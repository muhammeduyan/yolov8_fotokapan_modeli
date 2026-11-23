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
    def __init__(
        self,
        model_path=None,
        bank_path=None,
        threshold=None,
        percentile=5.0,
        device=None,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "model" / "best.pt"
        self.bank_path = Path(bank_path) if bank_path else PROJECT_ROOT / "model" / "knn_bank.npz"
        self.percentile = float(percentile)

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
        self.auto_threshold = 0.7000

        self._load_calibrated_bank()

        if threshold is None:
            self.threshold = self.auto_threshold
            self.is_auto_calibrated = True
        else:
            self.threshold = float(threshold)
            self.is_auto_calibrated = False

    @staticmethod
    def _l2_normalize(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8)

    def _extract_crop_raw_resnet(self, crop_bgr):
        """Kırpılan aday kutudan 2048 boyutlu ResNet50 öznitelik vektörü çıkarır."""
        resized_res = cv2.resize(crop_bgr, (224, 224))
        chw_res = resized_res[:, :, ::-1].transpose(2, 0, 1).copy()
        t_res = torch.from_numpy(chw_res).float().unsqueeze(0).to(self.device) / 255.0
        t_res = (t_res - self.im_mean) / self.im_std

        with torch.no_grad():
            vec_res = self.resnet_backbone(t_res).flatten(1).squeeze(0).cpu().numpy()

        return vec_res.astype(np.float32)

    def compute_calibrated_similarity(self, raw_vec):
        """
        Aday kutuyu:
        1. 1.811 Domuz Referans Bankası (U_tr) KNN benzerliği,
        2. 1.738 Kaya/Taş/Ağaç/Ot Arka Plan Referans Bankası (U_bg) KNN benzerliği,
        3. Ön Plan - Arka Plan Ayırıcı Vektör (w_fg) izdüşümü,
        4. 140-boyutlu Yaban Domuzu Alt-Uzayı (Vp) dikey izdüşüm hatası,
        5. Morfolojik Domuz (Sus scrofa) uyum skoru
        ile değerlendirip kalibre edilmiş [0, 1] benzerlik skorunu üretir.
        """
        if raw_vec.ndim == 1:
            R = raw_vec[np.newaxis, :]
        else:
            R = raw_vec
        U = self._l2_normalize(R)

        # 1. Domuz Referans Bankası KNN (k=5)
        sim_tr = U @ self.U_tr.T
        knn_boar = np.mean(np.partition(sim_tr, -self.k, axis=1)[:, -self.k :], axis=1)

        # 2. Kaya/Ağaç/Ot Arka Plan Bankası KNN (k=5)
        sim_bg = U @ self.U_bg.T
        knn_bg = np.mean(np.partition(sim_bg, -self.k, axis=1)[:, -self.k :], axis=1)

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
        calibrated_score = np.clip(self.auto_threshold + 0.35 * margin, 0.0, 0.999)
        return float(calibrated_score[0]) if calibrated_score.shape[0] == 1 else calibrated_score

    def _load_calibrated_bank(self):
        if not self.bank_path.exists():
            raise FileNotFoundError(f"Kalibre edilmiş referans bankası bulunamadı: {self.bank_path}")
        data = np.load(str(self.bank_path), allow_pickle=True)
        self.calibration_data = data
        self.U_tr = data["U_tr"]
        self.U_bg = data["U_bg"]
        self.mu = data["mu"]
        self.Vp = data["Vp"]
        self.w_fg = data["w_fg"]
        self.t1 = float(data["t1"])
        self.t2 = float(data["t2"])
        self.k = int(data["k"])
        self.auto_threshold = float(data["tau_95"])

    def get_calibration_summary(self):
        d = self.calibration_data
        s_tr = d["s_tr"]
        s_va = d["s_va"]
        s_te = d["s_te"]
        s_bg = d["s_bg"]
        tau = float(d["tau_95"])

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
            "train_mean": float(np.mean(s_tr)),
            "train_median": float(np.median(s_tr)),
            "val_mean": float(np.mean(s_va)),
            "val_median": float(np.median(s_va)),
            "val_p01": float(np.percentile(s_va, 1)),
            "val_p05": float(np.percentile(s_va, 5)),
            "val_p10": float(np.percentile(s_va, 10)),
            "val_tpr_count": int(np.sum(s_va >= tau)),
            "val_tpr_pct": float(np.mean(s_va >= tau) * 100.0),
            "test_mean": float(np.mean(s_te)),
            "test_median": float(np.median(s_te)),
            "test_tpr_count": int(np.sum(s_te >= tau)),
            "test_tpr_pct": float(np.mean(s_te >= tau) * 100.0),
            "bg_reject_pct": float(np.mean(s_bg < tau) * 100.0),
        }

    def predict(self, source_path, conf=0.25, save_dir=None):
        """
        Görüntü üzerinde çıkarım yapar; Domuz Bankası + Arka Plan (Kaya/Ağaç) Bankası +
        Sürü Bütünlüğü (Herd Context) kurallarına göre yanlış alarmları bastırır.
        """
        img_path = Path(str(source_path)).expanduser()
        if not img_path.exists() and str(source_path).lower().startswith("/desktop/"):
            img_path = Path.home() / "Desktop" / Path(str(source_path)).name
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Görsel açılamadı: {source_path} (Denenen yol: {img_path})")

        raw_output = img.copy()
        filtered_output = img.copy()

        results = self.yolo(img, conf=conf, verbose=False)
        boxes = results[0].boxes

        raw_detections = []
        filtered_detections = []
        suppressed_detections = []

        if len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().numpy().astype(int)
            conf_list = boxes.conf.cpu().numpy()

            candidates = []
            for box, c in zip(xyxy_list, conf_list):
                x1, y1, x2, y2 = box
                raw_detections.append({"box": box, "conf": float(c), "class": "wild_boar"})

                cv2.rectangle(raw_output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    raw_output,
                    f"wild_boar: {c:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                crop = img[max(0, y1) : min(img.shape[0], y2), max(0, x1) : min(img.shape[1], x2)]
                if crop.size == 0:
                    continue

                raw_vec = self._extract_crop_raw_resnet(crop)
                sim_score = self.compute_calibrated_similarity(raw_vec)
                candidates.append({"box": box, "conf": float(c), "sim": sim_score})

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
                c = cand["conf"]
                sim_score = cand["sim"]
                x1, y1, x2, y2 = box
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)

                if has_confirmed_lead_boar and c >= 0.50 and 0.56 <= sim_score < self.threshold:
                    sim_score = min(0.99, sim_score + 0.12)
                elif c >= 0.75 and (bw / bh) >= 2.2 and bh < 45 and 0.58 <= sim_score < self.threshold:
                    # Kare alt/üst sınırında veya tarih bandında yarısı kesilmiş yüksek güvenli domuz sırtı
                    sim_score = min(0.99, sim_score + 0.12)
                if sim_score >= self.threshold:
                    filtered_detections.append(
                        {"box": box, "conf": c, "sim": sim_score, "class": "wild_boar"}
                    )
                    cv2.rectangle(filtered_output, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        filtered_output,
                        f"wild_boar: {c:.2f} (Sim: {sim_score:.2f} >= {self.threshold:.2f})",
                        (x1, max(y1 - 10, 20)),
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
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            raw_file = out_dir / f"raw_model_a_{img_path.name}"
            filtered_file = out_dir / f"knn_filtered_{img_path.name}"

            cv2.imwrite(str(raw_file), raw_output)
            cv2.imwrite(str(filtered_file), filtered_output)

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
