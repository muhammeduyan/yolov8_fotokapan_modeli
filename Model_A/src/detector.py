"""
Model A (Local Wild Boar v0.4 - Tek Sınıflı Yaban Domuzu) için
Çift Bankalı (Domuz vs. Arka Plan) ve Morfolojik Alt-Uzay Kalibreli OOD Filtresi

Bu modül:
1. Eğitim setindeki 1.811 gerçek yaban domuzu kutusundan oluşturulan "Domuz Referans Bankası (U_tr)"
   ve 140-boyutlu PCA Alt-Uzayı (Vp) ile,
   domuz kutusu dışında kalan (IoU=0) taş, kaya, ağaç, çalı ve gece otlarından çıkarılan
   1.738 yamalık "Arka Plan Referans Bankası (U_bg)" ve Ayırıcı Vektörü (w_fg) kullanır.
2. Doğrulama (val) setindeki 457 gerçek yaban domuzu kutusu üzerinden eşik değerini (tau) otomatik kalibre eder.
3. Çıkarım sırasında her aday kutuyu Domuz Bankası, Arka Plan Bankası ve morfolojik domuz uyumuna göre
   değerlendirir; aynı karede kesinleşmiş (>=0.85) yetişkin domuz varsa sürü bireylerine destek uygular.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelAKNNFilter:
    """YOLOv8n tespitlerini ResNet-50 çift bankalı KNN ve PCA alt-uzayı ile filtreleyen sınıf."""

    def __init__(
        self,
        model_path=None,
        bank_path=None,
        threshold=None,
        percentile=5.0,
        device=None,
    ):
        """Model ağırlıklarını, ResNet-50 omurgasını ve kalibre edilmiş referans bankasını yükler;
        cihazı (CUDA/MPS/CPU) ve otomatik ya da manuel benzerlik eşiğini ayarlar."""
        self.percentile = float(percentile)
        if not (0.0 <= self.percentile <= 100.0):
            raise ValueError(f"percentile 0 ile 100 arasında olmalıdır: {percentile}")

        if device:
            dev_str = str(device).strip()
            self.device = f"cuda:{dev_str}" if dev_str.isdigit() else dev_str
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_path = (
            self._resolve_input_path(model_path) if model_path else PROJECT_ROOT / "model" / "best.pt"
        )
        self.bank_path = (
            self._resolve_input_path(bank_path) if bank_path else PROJECT_ROOT / "model" / "knn_bank.npz"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model dosyası bulunamadı: {self.model_path}")

        # 1. Local Wild Boar v0.4 (YOLOv8n - Close-Up + Arka Plan İnce Ayarlı) yükleniyor
        self.yolo = YOLO(str(self.model_path))

        # 2. Morfolojik ve doku öznitelik çıkarıcı (Dondurulmuş ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.resnet_backbone.eval()
        self.W_fc = resnet.fc.weight.detach().cpu().numpy().astype(np.float32)
        self.b_fc = resnet.fc.bias.detach().cpu().numpy().astype(np.float32)

        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 3. Kalibre edilmiş Domuz + Arka Plan (Kaya/Ağaç) referans bankaları yükleniyor
        self.calibration_data = None
        self.U_tr = None
        self.U_bg = None
        self.mu = None
        self.Vp = None
        self.w_fg = None
        self.t1 = 0.0
        self.t2 = 0.0
        self.k = 5
        self.score_offset = 0.7000
        self.auto_threshold = 0.7000

        self._load_calibrated_bank()

        if threshold is None:
            self.threshold = self.auto_threshold
            self.is_auto_calibrated = True
        else:
            self.threshold = float(threshold)
            self.is_auto_calibrated = False

    @staticmethod
    def _resolve_input_path(path):
        """Göreli girdi yolunu önce mevcut çalışma dizininde, yoksa PROJECT_ROOT altında arar."""
        p = Path(str(path)).expanduser()
        if not p.is_absolute() and not p.exists():
            candidate = PROJECT_ROOT / p
            if candidate.exists():
                return candidate
        return p

    @staticmethod
    def _resolve_save_dir(save_dir):
        """Çıktı klasörünü mutlak yol, mevcut dizin veya PROJECT_ROOT yapısına göre çözümler."""
        out_dir = Path(str(save_dir)).expanduser()
        if out_dir.is_absolute() or out_dir.exists():
            return out_dir
        if (PROJECT_ROOT / out_dir).exists() or (
            len(out_dir.parts) > 1
            and (PROJECT_ROOT / out_dir.parts[0]).exists()
            and not Path(out_dir.parts[0]).exists()
        ):
            return PROJECT_ROOT / out_dir
        return out_dir

    @staticmethod
    def _l2_normalize(x):
        """Verilen öznitelik vektörünü veya matrisini son eksen boyunca L2 birim uzunluğa normalize eder."""
        x_arr = np.asarray(x, dtype=np.float32)
        norm = np.linalg.norm(x_arr, axis=-1, keepdims=True)
        return x_arr / (norm + 1e-8)

    def _extract_crop_raw_resnet(self, crop_bgr):
        """BGR formatındaki aday kutu kırpıntısını 224x224 boyutuna getirip 2048 boyutlu ResNet-50 vektörü döndürür."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Kırpılan görsel bölgesi boş olamaz.")
        resized_res = cv2.resize(crop_bgr, (224, 224))
        chw_res = resized_res[:, :, ::-1].transpose(2, 0, 1).copy()
        t_res = torch.from_numpy(chw_res).float().unsqueeze(0).to(self.device) / 255.0
        t_res = (t_res - self.im_mean) / self.im_std

        with torch.no_grad():
            vec_res = self.resnet_backbone(t_res).flatten(1).squeeze(0).cpu().numpy()

        return vec_res.astype(np.float32)

    def compute_calibrated_similarity(self, raw_vec):
        """2048 boyutlu ResNet-50 vektörünü çift bankalı KNN, PCA alt-uzayı ve morfolojik uyumla
        değerlendirerek [0, 0.999] aralığında kalibre edilmiş domuz benzerlik skoru (float veya ndarray) döndürür."""
        vec = np.asarray(raw_vec, dtype=np.float32)
        if vec.ndim == 1:
            R = vec[np.newaxis, :]
        elif vec.ndim == 2:
            R = vec
        else:
            raise ValueError(f"raw_vec 1B veya 2B olmalıdır, alınan boyut: {vec.ndim}")

        if R.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        U = self._l2_normalize(R)

        # 1. Domuz Referans Bankası KNN
        sim_tr = U @ self.U_tr.T
        k_tr = min(self.k, self.U_tr.shape[0])
        knn_boar = np.mean(np.partition(sim_tr, -k_tr, axis=1)[:, -k_tr:], axis=1)

        # 2. Kaya/Ağaç/Ot Arka Plan Bankası KNN
        sim_bg = U @ self.U_bg.T
        k_bg = min(self.k, self.U_bg.shape[0])
        knn_bg = np.mean(np.partition(sim_bg, -k_bg, axis=1)[:, -k_bg:], axis=1)

        # 3. Ön Plan (Domuz) vs Arka Plan (Kaya/Ağaç) Ayırıcı İzdüşüm
        fg_proj = U @ self.w_fg

        # 4. Domuz Alt-Uzayı Dikey Mesafe Hatası (Subspace Residual)
        Uc = U - self.mu
        res = np.linalg.norm(Uc - (Uc @ self.Vp.T) @ self.Vp, axis=1)

        # 5. Morfolojik Domuz Uyumu
        logits = R @ self.W_fc.T + self.b_fc
        swine_logit = np.max(logits[:, [341, 342, 343]], axis=1)
        other_logits = np.delete(logits, [341, 342, 343], axis=1)
        max_other = np.max(other_logits, axis=1)
        swine_margin = swine_logit - max_other

        sw_rank = np.sum(logits > swine_logit[:, None], axis=1) + 1
        top3_swine_bonus = np.where(sw_rank <= 3, 0.22, 0.0)

        # Kesin Ayı (294-297), Kurt/Çakal (269-275) veya Fil (101, 385, 386) Morfolojik Cezası
        top1_cls = np.argmax(logits, axis=1)
        ood_predator_mask = np.isin(
            top1_cls, [101, 269, 270, 271, 272, 273, 274, 275, 294, 295, 296, 297, 385, 386]
        ) & (sw_rank > 5)
        ood_penalty = np.where(ood_predator_mask, -0.08, 0.0)

        # Kapı 1 (OOD Hayvan & Alt-Uzay Kapısı) ve Kapı 2 (Kaya/Ağaç Arka Plan Kapısı)
        g1 = (
            knn_boar
            - 0.35 * knn_bg
            - 1.90 * res
            + 0.035 * np.clip(swine_margin, -6.0, 5.0)
            + top3_swine_bonus
            + ood_penalty
        )
        g2 = (
            (knn_boar - knn_bg)
            + 0.80 * fg_proj
            + 0.040 * np.clip(swine_margin, -6.0, 5.0)
            + top3_swine_bonus
        )

        margin = np.minimum(g1 - self.t1, (g2 - self.t2) * 0.7)
        calibrated_score = np.clip(self.score_offset + 0.35 * margin, 0.0, 0.999)
        return float(calibrated_score[0]) if calibrated_score.shape[0] == 1 else calibrated_score

    def _load_calibrated_bank(self):
        """Referans .npz dosyasından bankaları, PCA alt-uzayını ve kalibrasyon parametrelerini yükler;
        seçilen percentile değerine göre auto_threshold değerini ayarlar."""
        if not self.bank_path.exists():
            raise FileNotFoundError(f"Kalibre edilmiş referans bankası bulunamadı: {self.bank_path}")
        with np.load(str(self.bank_path), allow_pickle=True) as data:
            self.calibration_data = {k: data[k] for k in data.files}

        d = self.calibration_data
        self.U_tr = d["U_tr"]
        self.U_bg = d["U_bg"]
        self.mu = d["mu"]
        self.Vp = d["Vp"]
        self.w_fg = d["w_fg"]
        self.t1 = float(d["t1"])
        self.t2 = float(d["t2"])
        self.k = int(d["k"])

        if len(self.U_tr) == 0 or len(self.U_bg) == 0:
            raise ValueError("Referans bankaları (U_tr ve U_bg) boş olamaz.")
        if self.k < 1:
            raise ValueError(f"KNN k değeri en az 1 olmalıdır: {self.k}")

        s_va = d["s_va"]
        if len(s_va) == 0:
            raise ValueError("Doğrulama benzerlik dizisi (s_va) boş olamaz.")

        self.score_offset = float(d["tau_95"]) if "tau_95" in d else float(np.percentile(s_va, 5.0))
        if np.isclose(self.percentile, 5.0) and "tau_95" in d:
            self.auto_threshold = self.score_offset
        else:
            self.auto_threshold = float(np.percentile(s_va, self.percentile))

    def get_calibration_summary(self):
        """Kalibrasyon bankasındaki eğitim, doğrulama, test ve arka plan istatistiklerini
        özetleyen bir sözlük (dict) döndürür."""
        d = self.calibration_data
        s_tr = d["s_tr"]
        s_va = d["s_va"]
        s_te = d["s_te"]
        s_bg = d["s_bg"]
        tau = self.auto_threshold

        return {
            "model_path": str(self.model_path),
            "bank_path": str(self.bank_path),
            "n_train_imgs": int(d["n_train_imgs"]),
            "n_train_boxes": int(d["n_train_boxes"]),
            "n_bg_patches": int(len(self.U_bg)),
            "n_val_imgs": int(d["n_val_imgs"]),
            "n_val_boxes": int(d["n_val_boxes"]),
            "n_test_imgs": int(d["n_test_imgs"]),
            "n_test_boxes": int(d["n_test_boxes"]),
            "percentile": self.percentile,
            "target_val_tpr": 100.0 - self.percentile,
            "calibrated_tau": tau,
            "train_mean": float(np.mean(s_tr)) if len(s_tr) > 0 else 0.0,
            "train_median": float(np.median(s_tr)) if len(s_tr) > 0 else 0.0,
            "val_mean": float(np.mean(s_va)),
            "val_median": float(np.median(s_va)),
            "val_p01": float(np.percentile(s_va, 1)),
            "val_p05": float(np.percentile(s_va, 5)),
            "val_p10": float(np.percentile(s_va, 10)),
            "val_tpr_count": int(np.sum(s_va >= tau)),
            "val_tpr_pct": float(np.mean(s_va >= tau) * 100.0),
            "test_mean": float(np.mean(s_te)) if len(s_te) > 0 else 0.0,
            "test_median": float(np.median(s_te)) if len(s_te) > 0 else 0.0,
            "test_tpr_count": int(np.sum(s_te >= tau)) if len(s_te) > 0 else 0,
            "test_tpr_pct": float(np.mean(s_te >= tau) * 100.0) if len(s_te) > 0 else 0.0,
            "bg_reject_pct": float(np.mean(s_bg < tau) * 100.0) if len(s_bg) > 0 else 0.0,
        }

    def predict(self, source_path, conf=0.25, save_dir=None):
        """Verilen görselde YOLO ile aday domuzları tespit eder, KNN/OOD filtresi ve sürü bağlamı
        kurallarıyla doğrular; ham/filtreli tespit listeleri ile kayıt yollarını içeren sözlük döndürür."""
        img_path = self._resolve_input_path(source_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Görsel bulunamadı: {source_path} (Denenen yol: {img_path})")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Görsel dosyası okunamadı veya bozuk: {img_path}")

        h, w = img.shape[:2]
        raw_output = img.copy()
        filtered_output = img.copy()

        results = self.yolo(img, conf=conf, device=self.device, verbose=False)
        boxes = results[0].boxes

        raw_detections = []
        filtered_detections = []
        suppressed_detections = []

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            conf_list = boxes.conf.cpu().numpy()

            candidates = []
            for box, c in zip(xyxy_list, conf_list):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx1 = max(0, min(w, x1))
                cy1 = max(0, min(h, y1))
                cx2 = max(0, min(w, x2))
                cy2 = max(0, min(h, y2))

                raw_detections.append({"box": box, "conf": float(c), "class": "wild_boar"})

                cv2.rectangle(raw_output, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
                cv2.putText(
                    raw_output,
                    f"wild_boar: {c:.2f}",
                    (cx1, max(cy1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    suppressed_detections.append(
                        {"box": box, "conf": float(c), "sim": 0.0, "class": "OOD_suppressed"}
                    )
                    continue

                raw_vec = self._extract_crop_raw_resnet(crop)
                sim_score = self.compute_calibrated_similarity(raw_vec)
                candidates.append(
                    {
                        "box": box,
                        "clamped": (cx1, cy1, cx2, cy2),
                        "conf": float(c),
                        "sim": sim_score,
                    }
                )

            # Sürü Bütünlüğü Kuralı (Herd Context):
            # Aynı karede çok yüksek benzerlikli (>= 0.85) kesinleşmiş bir yetişkin domuz
            # VEYA en az 2 adet onaylanmış domuz (sim >= threshold, conf >= 0.50) varsa,
            # sürüdeki diğer bireylere (conf >= 0.50 ve 0.56 <= sim < threshold) +0.12 destek uygulanır.
            n_confirmed_boars = sum(
                1 for cand in candidates if cand["sim"] >= self.threshold and cand["conf"] >= 0.50
            )
            has_confirmed_lead_boar = (
                any(cand["sim"] >= 0.85 and cand["conf"] >= 0.75 for cand in candidates)
                or n_confirmed_boars >= 2
            )

            for cand in candidates:
                box = cand["box"]
                cx1, cy1, cx2, cy2 = cand["clamped"]
                c = cand["conf"]
                sim_score = cand["sim"]
                bw, bh = max(1, cx2 - cx1), max(1, cy2 - cy1)

                if has_confirmed_lead_boar and c >= 0.50 and 0.56 <= sim_score < self.threshold:
                    sim_score = min(0.99, sim_score + 0.12)
                elif c >= 0.75 and (bw / bh) >= 2.2 and bh < 45 and 0.58 <= sim_score < self.threshold:
                    # Kare alt/üst sınırında veya tarih bandında yarısı kesilmiş yüksek güvenli domuz sırtı
                    sim_score = min(0.99, sim_score + 0.12)
                if sim_score >= self.threshold:
                    filtered_detections.append(
                        {"box": box, "conf": c, "sim": sim_score, "class": "wild_boar"}
                    )
                    cv2.rectangle(filtered_output, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
                    cv2.putText(
                        filtered_output,
                        f"wild_boar: {c:.2f} (Sim: {sim_score:.2f} >= {self.threshold:.2f})",
                        (cx1, max(cy1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                else:
                    suppressed_detections.append(
                        {"box": box, "conf": c, "sim": sim_score, "class": "OOD_suppressed"}
                    )

        saved_files = {}
        if save_dir:
            out_dir = self._resolve_save_dir(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            raw_file = out_dir / f"raw_model_a_{img_path.name}"
            filtered_file = out_dir / f"knn_filtered_{img_path.name}"

            if not cv2.imwrite(str(raw_file), raw_output):
                raise OSError(f"Ham çıktı görseli kaydedilemedi: {raw_file}")
            if not cv2.imwrite(str(filtered_file), filtered_output):
                raise OSError(f"Filtreli çıktı görseli kaydedilemedi: {filtered_file}")

            saved_files["raw"] = str(raw_file)
            saved_files["filtered"] = str(filtered_file)

        return {
            "source": str(img_path),
            "threshold": self.threshold,
            "is_auto_calibrated": self.is_auto_calibrated,
            "raw_detections": raw_detections,
            "filtered_detections": filtered_detections,
            "suppressed_detections": suppressed_detections,
            "saved_files": saved_files,
        }
