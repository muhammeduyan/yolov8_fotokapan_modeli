"""
Model A (Local Wild Boar v0.4 - Tek Sınıflı Yaban Domuzu) için
Temiz ResNet-50 Öznitelik Uzayı Tabanlı Çift Bankalı OOD Filtresi.

Bu modül:
1. Eğitim setindeki gerçek yaban domuzu kutularından oluşturulan Domuz Referans Bankası (U_tr),
   %90 varyansı açıklayan PCA Alt-Uzayı (Vp) ve OAS (Oracle Approximating Shrinkage) büzülmeli
   Mahalanobis uzaklığı ile birlikte, domuz dışı (IoU=0) Arka Plan Referans Bankasını (U_bg) kullanır.
2. Tüm aday kutuları %10 bağlam paylı aspect-ratio koruyucu Letterbox kırpma (extract_letterbox_crop)
   ile 224x224 boyutuna getirerek saf 2048 boyutlu ResNet-50 öznitelik uzayında değerlendirir.
3. Eğitim setinde öğrenilen L2 Lojistik Regresyon ağırlıkları ve doğrulama (val) setinden kalibre
   edilen eşik değeri (tau) ile karar verir; sıfır elle ayarlanmış sabit ve sıfır ImageNet sınıf kuralı içerir.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_letterbox_crop(
    img_bgr,
    box_xyxy,
    pad_ratio=0.10,
    out_size=224,
    pad_color=(114, 114, 114),
):
    """Verilen sınırlayıcı kutuyu her yönde pad_ratio oranında genişletir, görüntü sınırlarına
    kırpar, en-boy oranını koruyarak out_size boyutuna ölçekler ve pad_color ile ortalayarak doldurur."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Girdi görseli boş olamaz.")

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    pad_x = float(pad_ratio) * bw
    pad_y = float(pad_ratio) * bh
    ex1 = max(0, int(np.floor(x1 - pad_x)))
    ey1 = max(0, int(np.floor(y1 - pad_y)))
    ex2 = min(w, int(np.ceil(x2 + pad_x)))
    ey2 = min(h, int(np.ceil(y2 + pad_y)))

    if ex2 <= ex1 or ey2 <= ey1:
        raise ValueError("Genişletilmiş kırpma kutusu boş veya geçersiz.")

    crop = img_bgr[ey1:ey2, ex1:ex2]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        raise ValueError("Kırpılan görsel bölgesi boş olamaz.")

    scale = float(out_size) / float(max(ch, cw))
    if cw >= ch:
        new_w = int(out_size)
        new_h = max(1, min(int(out_size), int(round(ch * scale))))
    else:
        new_h = int(out_size)
        new_w = max(1, min(int(out_size), int(round(cw * scale))))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    canvas = np.full((int(out_size), int(out_size), 3), pad_color, dtype=crop.dtype)
    top = (int(out_size) - new_h) // 2
    left = (int(out_size) - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def l2_normalize(x):
    """Verilen öznitelik vektörünü veya matrisini son eksen boyunca L2 birim uzunluğa normalize eder."""
    x_arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x_arr, axis=-1, keepdims=True)
    return x_arr / (norm + 1e-8)


def sigmoid(x):
    """Sayısal taşmayı önleyerek girdi dizisi veya skaler üzerinde sigmoid dönüşümü uygular."""
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_knn_cosine_mean(U, U_bank, k=5, exclude_self=False):
    """L2-normalize öznitelikler ile referans bankası arasındaki en yüksek k kosinüs benzerliğinin ortalamasını döndürür."""
    sim = np.asarray(U, dtype=np.float32) @ np.asarray(U_bank, dtype=np.float32).T
    if exclude_self:
        if sim.shape[0] != sim.shape[1]:
            raise ValueError("exclude_self=True iken U ve U_bank aynı boyutta olmalıdır.")
        np.fill_diagonal(sim, -np.inf)
        k_eff = min(int(k), U_bank.shape[0] - 1)
    else:
        k_eff = min(int(k), U_bank.shape[0])

    if k_eff < 1:
        raise ValueError(f"Geçerli KNN komşu sayısı en az 1 olmalıdır: {k_eff}")

    top_k = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    return np.mean(top_k, axis=1).astype(np.float64)


def compute_pca_residual(Uc, Vp):
    """Merkezlenmiş özniteliklerin PCA alt-uzayına dik kalan L2 artık uzaklığını hesaplar."""
    Uc64 = np.asarray(Uc, dtype=np.float64)
    Vp64 = np.asarray(Vp, dtype=np.float64)
    proj = (Uc64 @ Vp64.T) @ Vp64
    return np.linalg.norm(Uc64 - proj, axis=1)


def compute_oas_maha_dist(Uc, maha_V, maha_eigvals, maha_alpha, maha_beta):
    """OAS büzülmeli kovaryans matrisinin öz-ayrışımı üzerinden tam Mahalanobis uzaklığını hesaplar."""
    Uc64 = np.asarray(Uc, dtype=np.float64)
    V64 = np.asarray(maha_V, dtype=np.float64)
    eig64 = np.asarray(maha_eigvals, dtype=np.float64)
    alpha = float(maha_alpha)
    beta = max(float(maha_beta), 1e-12)

    inv_eig = 1.0 / ((1.0 - alpha) * eig64 + beta)
    proj = Uc64 @ V64.T
    norm_sq = np.sum(Uc64 ** 2, axis=1)
    proj_sq = np.sum(proj ** 2, axis=1)
    orth_sq = np.maximum(0.0, norm_sq - proj_sq)
    maha_sq = np.sum((proj ** 2) * inv_eig, axis=1) + orth_sq / beta
    return np.sqrt(np.maximum(0.0, maha_sq))


def extract_ood_signals(
    U,
    U_tr,
    U_bg,
    mu,
    Vp,
    maha_V,
    maha_eigvals,
    maha_alpha,
    maha_beta,
    k=5,
    exclude_self_tr=False,
    exclude_self_bg=False,
):
    """L2-normalize ResNet-50 özniteliklerinden 4 saf OOD sinyalini ([knn_boar, -knn_bg, -pca_residual, -maha_dist]) üretir."""
    knn_boar = compute_knn_cosine_mean(U, U_tr, k=k, exclude_self=exclude_self_tr)
    knn_bg = compute_knn_cosine_mean(U, U_bg, k=k, exclude_self=exclude_self_bg)
    Uc = np.asarray(U, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    pca_res = compute_pca_residual(Uc, Vp)
    maha_dist = compute_oas_maha_dist(Uc, maha_V, maha_eigvals, maha_alpha, maha_beta)
    return np.column_stack([knn_boar, -knn_bg, -pca_res, -maha_dist])


def compute_dual_gate_margin(
    phi,
    ood_mean,
    ood_std,
    ood_val_tau,
    ood_val_scale,
    feat_mean,
    feat_std,
    weights,
    bias,
    bg_val_tau,
    bg_val_scale,
):
    """Bağımsız Çift Kapı (Kapı 1: Saf Tek-Sınıflı Domuz OOD ∧ Kapı 2: Arka Plan) birleşik marjını hesaplar.

    Kapı 1 yalnızca domuz dağılımı sinyallerini ([knn_boar, -pca_residual, -maha_dist]) eşit ağırlıkla
    kullanır ve arka plan bankasına bakmaz; Kapı 2 ise domuz-arka plan ayrımı yapan lojistik skordur.
    Her iki kapı doğrulama seti ölçeğine getirilip minimum (mantıksal VE) ile birleştirilir.
    """
    phi_ood = phi[:, [0, 2, 3]]
    z_ood = np.mean((phi_ood - ood_mean) / ood_std, axis=1)
    m_ood = (z_ood - ood_val_tau) / ood_val_scale

    z_bg = ((phi - feat_mean) / feat_std) @ weights + bias
    m_bg = (z_bg - bg_val_tau) / bg_val_scale

    return np.minimum(m_ood, m_bg), m_ood, m_bg


class ModelAKNNFilter:
    """YOLOv8n tespitlerini saf ResNet-50 öznitelik uzayında çift bankalı KNN, PCA alt-uzayı
    ve OAS Mahalanobis uzaklığı ile filtreleyen sınıf."""

    def __init__(
        self,
        model_path=None,
        bank_path=None,
        threshold=None,
        percentile=5.0,
        device=None,
    ):
        """Model ağırlıklarını, dondurulmuş ResNet-50 omurgasını ve kalibre edilmiş referans
        bankasını yükler; cihazı ve otomatik ya da manuel benzerlik eşiğini ayarlar."""
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

        # 1. Local Wild Boar v0.4 (YOLOv8n) yükleniyor
        self.yolo = YOLO(str(self.model_path))

        # 2. Saf ResNet-50 öznitelik çıkarıcı (sadece 2048-D omurga, sınıf logitleri hariç)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.resnet_backbone.eval()

        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 3. Kalibre edilmiş referans bankası ve öğrenilmiş OOD parametreleri
        self.calibration_data = None
        self.U_tr = None
        self.U_bg = None
        self.mu = None
        self.Vp = None
        self.maha_V = None
        self.maha_eigvals = None
        self.maha_alpha = 0.0
        self.maha_beta = 1.0
        self.ood_mean = None
        self.ood_std = None
        self.ood_val_tau = 0.0
        self.ood_val_scale = 1.0
        self.feat_mean = None
        self.feat_std = None
        self.weights = None
        self.bias = 0.0
        self.bg_val_tau = 0.0
        self.bg_val_scale = 1.0
        self.tau_raw_95 = 0.0
        self.score_scale = 1.0
        self.pad_ratio = 0.10
        self.k = 5
        self.auto_threshold = 0.5000

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
        return l2_normalize(x)

    def _extract_crop_raw_resnet(self, crop_bgr):
        """224x224 (veya ham BGR) kırpıntıdan 2048 boyutlu ResNet-50 öznitelik vektörü döndürür."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Kırpılan görsel bölgesi boş olamaz.")

        if crop_bgr.shape[:2] != (224, 224):
            ch, cw = crop_bgr.shape[:2]
            interp = cv2.INTER_AREA if max(ch, cw) > 224 else cv2.INTER_CUBIC
            resized_res = cv2.resize(crop_bgr, (224, 224), interpolation=interp)
        else:
            resized_res = crop_bgr

        chw_res = resized_res[:, :, ::-1].transpose(2, 0, 1).copy()
        t_res = torch.from_numpy(chw_res).float().unsqueeze(0).to(self.device) / 255.0
        t_res = (t_res - self.im_mean) / self.im_std

        with torch.no_grad():
            vec_res = self.resnet_backbone(t_res).flatten(1).squeeze(0).cpu().numpy()

        return vec_res.astype(np.float32)

    def compute_calibrated_similarity(self, raw_vec):
        """2048 boyutlu ResNet-50 vektörünü 4 saf öznitelik-uzayı OOD sinyali ve öğrenilmiş
        L2 lojistik regresyon ağırlıklarıyla değerlendirip (0, 1) aralığında domuz skoru döndürür."""
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
        phi = extract_ood_signals(
            U,
            self.U_tr,
            self.U_bg,
            self.mu,
            self.Vp,
            self.maha_V,
            self.maha_eigvals,
            self.maha_alpha,
            self.maha_beta,
            k=self.k,
        )
        z_joint, _, _ = compute_dual_gate_margin(
            phi,
            self.ood_mean,
            self.ood_std,
            self.ood_val_tau,
            self.ood_val_scale,
            self.feat_mean,
            self.feat_std,
            self.weights,
            self.bias,
            self.bg_val_tau,
            self.bg_val_scale,
        )
        calibrated_score = sigmoid((z_joint - self.tau_raw_95) / self.score_scale).astype(np.float32)
        return float(calibrated_score[0]) if calibrated_score.shape[0] == 1 else calibrated_score

    def _load_calibrated_bank(self):
        """Referans .npz dosyasından bankaları, PCA/OAS-Mahalanobis parametrelerini ve bağımsız
        çift kapı parametrelerini yükler; seçilen percentile değerine göre auto_threshold hesaplar."""
        if not self.bank_path.exists():
            raise FileNotFoundError(f"Kalibre edilmiş referans bankası bulunamadı: {self.bank_path}")
        with np.load(str(self.bank_path), allow_pickle=False) as data:
            self.calibration_data = {k: data[k] for k in data.files}

        d = self.calibration_data
        self.U_tr = d["U_tr"].astype(np.float32)
        self.U_bg = d["U_bg"].astype(np.float32)
        self.mu = d["mu"].astype(np.float64)
        self.Vp = d["Vp"].astype(np.float64)
        self.maha_V = d["maha_V"].astype(np.float64)
        self.maha_eigvals = d["maha_eigvals"].astype(np.float64)
        self.maha_alpha = float(d["maha_alpha"])
        self.maha_beta = float(d["maha_beta"])
        self.ood_mean = d["ood_mean"].astype(np.float64)
        self.ood_std = d["ood_std"].astype(np.float64)
        self.ood_val_tau = float(d["ood_val_tau"])
        self.ood_val_scale = max(float(d["ood_val_scale"]), 1e-3)
        self.feat_mean = d["feat_mean"].astype(np.float64)
        self.feat_std = d["feat_std"].astype(np.float64)
        self.weights = d["weights"].astype(np.float64)
        self.bias = float(d["bias"])
        self.bg_val_tau = float(d["bg_val_tau"])
        self.bg_val_scale = max(float(d["bg_val_scale"]), 1e-3)
        self.tau_raw_95 = float(d["tau_raw_95"])
        self.score_scale = max(float(d["score_scale"]), 1e-3)
        self.pad_ratio = float(d["pad_ratio"])
        self.k = int(d["k"])

        if len(self.U_tr) == 0 or len(self.U_bg) == 0:
            raise ValueError("Referans bankaları (U_tr ve U_bg) boş olamaz.")
        if self.k < 1:
            raise ValueError(f"KNN k değeri en az 1 olmalıdır: {self.k}")

        s_va = d["s_va"]
        if len(s_va) == 0:
            raise ValueError("Doğrulama benzerlik dizisi (s_va) boş olamaz.")

        if np.isclose(self.percentile, 5.0):
            self.auto_threshold = 0.5000
        else:
            self.auto_threshold = float(np.percentile(s_va, self.percentile))

    def get_calibration_summary(self):
        """Kalibrasyon bankasındaki eğitim, doğrulama, test, arka plan istatistiklerini ve
        öğrenilen model parametrelerini özetleyen bir sözlük (dict) döndürür."""
        d = self.calibration_data
        s_tr = d["s_tr"]
        s_va = d["s_va"]
        s_te = d["s_te"]
        s_bg = d["s_bg"]
        tau = self.auto_threshold

        return {
            "model_path": str(self.model_path),
            "bank_path": str(self.bank_path),
            "pad_ratio": float(self.pad_ratio),
            "k": int(self.k),
            "pca_dim": int(self.Vp.shape[0]),
            "pca_var_ratio": float(d["pca_var_ratio"]),
            "shrinkage_alpha": float(self.maha_alpha),
            "l2_reg": float(d["l2_reg"]),
            "feature_names": ["knn_boar", "-knn_bg", "-pca_residual", "-maha_dist"],
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "tau_raw_95": float(self.tau_raw_95),
            "score_scale": float(self.score_scale),
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
            "bg_mean": float(np.mean(s_bg)) if len(s_bg) > 0 else 0.0,
            "bg_median": float(np.median(s_bg)) if len(s_bg) > 0 else 0.0,
            "bg_reject_pct": float(np.mean(s_bg < tau) * 100.0) if len(s_bg) > 0 else 0.0,
        }

    def predict(self, source_path, conf=0.25, save_dir=None):
        """Verilen görselde YOLO ile aday domuzları tespit eder, her kutuyu Letterbox kırpma ve
        saf ResNet-50 OOD filtresi ile doğrular; ham ve filtreli sonuç sözlüğü döndürür."""
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

                if cx2 <= cx1 or cy2 <= cy1:
                    suppressed_detections.append(
                        {"box": box, "conf": float(c), "sim": 0.0, "class": "OOD_suppressed"}
                    )
                    continue

                crop = extract_letterbox_crop(img, (cx1, cy1, cx2, cy2), pad_ratio=self.pad_ratio)
                raw_vec = self._extract_crop_raw_resnet(crop)
                sim_score = self.compute_calibrated_similarity(raw_vec)

                if sim_score >= self.threshold:
                    filtered_detections.append(
                        {"box": box, "conf": float(c), "sim": sim_score, "class": "wild_boar"}
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
                        {"box": box, "conf": float(c), "sim": sim_score, "class": "OOD_suppressed"}
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
