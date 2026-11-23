#!/usr/bin/env python3
"""
build_bank.py: Sıfır sızıntılı, %10 bağlam paylı Letterbox + saf ResNet-50 öznitelik uzayı
referans bankası (knn_bank.npz) oluşturucu.

Bu modül:
1. Yalnızca orijinal eğitim görsellerinden (*_closeup* ve *_bg* hariç) gerçek yaban domuzu
   kutularını (U_tr) ve aynı eğitim görsellerinden deterministik (seed=42) IoU=0 arka plan
   yamalarını (U_bg) çıkarır.
2. Eğitim öznitelikleri üzerinde PCA (%90 varyans), OAS (Chen et al. 2010) kovaryans
   büzülmesi ve saf NumPy Newton-IRLS L2 Lojistik Regresyon parametrelerini öğrenir.
3. Eşik (tau_raw_95) ve ölçeği (score_scale) yalnızca doğrulama (val) setinden kalibre eder;
   test setini yalnızca salt-okunur özet istatistikleri için değerlendirir.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .detector import (
        PROJECT_ROOT,
        extract_letterbox_crop,
        extract_ood_signals,
        l2_normalize,
        sigmoid,
    )
except ImportError:
    try:
        from src.detector import (
            PROJECT_ROOT,
            extract_letterbox_crop,
            extract_ood_signals,
            l2_normalize,
            sigmoid,
        )
    except ImportError:
        from detector import (
            PROJECT_ROOT,
            extract_letterbox_crop,
            extract_ood_signals,
            l2_normalize,
            sigmoid,
        )

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_device(device=None):
    """Çıkarım için en uygun donanımı (CUDA, MPS veya CPU) seçer."""
    if device:
        dev_str = str(device).strip()
        return f"cuda:{dev_str}" if dev_str.isdigit() else dev_str
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_split_images(img_dir, exclude_synthetic=False):
    """Verilen split klasöründeki görselleri deterministik sırada listeler; istenirse sentetik dosyaları dışlar."""
    if not img_dir.exists():
        raise FileNotFoundError(f"Görsel klasörü bulunamadı: {img_dir}")
    paths = []
    for p in sorted(img_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if exclude_synthetic and ("_closeup" in p.stem or "_bg" in p.stem):
            continue
        paths.append(p)
    return paths


def read_yolo_boxes_xyxy(lbl_path, w, h):
    """YOLO normalize etiket dosyasından piksel koordinatlı [x1, y1, x2, y2] kutu listesi okur."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = max(0.0, (cx - bw / 2.0) * w)
        y1 = max(0.0, (cy - bh / 2.0) * h)
        x2 = min(float(w), (cx + bw / 2.0) * w)
        y2 = min(float(h), (cy + bh / 2.0) * h)
        if x2 > x1 + 1.0 and y2 > y1 + 1.0:
            boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_overlap(box_a, box_b):
    """İki [x1, y1, x2, y2] kutusunun kesişip kesişmediğini döndürür."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def expand_box(box_xyxy, pad_ratio, w, h):
    """Kutuyu pad_ratio oranında genişletir ve görüntü sınırlarına kırpar."""
    x1, y1, x2, y2 = box_xyxy
    bw = x2 - x1
    bh = y2 - y1
    return (
        max(0.0, x1 - pad_ratio * bw),
        max(0.0, y1 - pad_ratio * bh),
        min(float(w), x2 + pad_ratio * bw),
        min(float(h), y2 + pad_ratio * bh),
    )


def collect_gt_crops(data_root, split, pad_ratio=0.10, exclude_synthetic=False):
    """Belirtilen veri kümesi bölümündeki tüm GT kutuları için 224x224 Letterbox kırpıntılarını toplar."""
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    img_paths = list_split_images(img_dir, exclude_synthetic=exclude_synthetic)

    crops = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        for box in read_yolo_boxes_xyxy(lbl_path, w, h):
            crops.append(extract_letterbox_crop(img, box, pad_ratio=pad_ratio))

    return len(img_paths), crops


def box_iou(box_a, box_b):
    """İki [x1, y1, x2, y2] kutusunun IoU (Intersection over Union) oranını hesaplar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def collect_background_crops(
    data_root,
    train_img_paths,
    pad_ratio=0.10,
    target_count=1800,
    seed=42,
):
    """Orijinal eğitim görsellerinden GT kutularıyla IoU=0 olan, fotokapan şeritleri dışında
    dokulu arka plan yamalarını deterministik olarak toplar."""
    lbl_dir = data_root / "labels" / "train"
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(train_img_paths))
    max_per_image = 8

    per_image_crops = []
    for idx in order:
        img_path = train_img_paths[int(idx)]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        y_top = int(np.ceil(h * 0.06))
        y_bot = int(np.floor(h * 0.91))
        if y_bot - y_top < 32:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        raw_gt = read_yolo_boxes_xyxy(lbl_path, w, h)
        # Hem GT kutusuna güvenlik payı hem de yama genişlemesine karşı tam IoU=0 koruması
        safe_gt = [expand_box(b, pad_ratio + 0.05, w, h) for b in raw_gt]

        img_crops = []
        img_boxes = []
        for _ in range(50):
            if len(img_crops) >= max_per_image:
                break
            pw = int(rng.integers(max(16, int(w * 0.08)), max(17, int(w * 0.30) + 1)))
            aspect = float(rng.uniform(0.65, 1.45))
            ph = max(16, int(round(pw * aspect)))

            pad_x = int(np.ceil(pad_ratio * pw))
            pad_y = int(np.ceil(pad_ratio * ph))
            if pw + 2 * pad_x >= w or ph + 2 * pad_y >= (y_bot - y_top):
                continue

            px1 = int(rng.integers(pad_x, w - pw - pad_x + 1))
            py1 = int(rng.integers(y_top + pad_y, y_bot - ph - pad_y + 1))
            px2 = px1 + pw
            py2 = py1 + ph

            p_box = (px1, py1, px2, py2)
            expanded_p = (px1 - pad_x, py1 - pad_y, px2 + pad_x, py2 + pad_y)
            if any(boxes_overlap(expanded_p, gb) for gb in safe_gt):
                continue
            if any(box_iou(p_box, prev_b) > 0.25 for prev_b in img_boxes):
                continue

            patch = img[py1:py2, px1:px2]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) > 20.0 and float(cv2.Laplacian(gray, cv2.CV_64F).var()) > 20.0:
                img_crops.append(extract_letterbox_crop(img, p_box, pad_ratio=pad_ratio))
                img_boxes.append(p_box)

        if img_crops:
            per_image_crops.append(img_crops)

    # Tüm görsellere dengeli yayılım için tur bazlı (round-robin) toplama
    bg_crops = []
    for round_idx in range(max_per_image):
        for plist in per_image_crops:
            if round_idx < len(plist):
                bg_crops.append(plist[round_idx])
                if len(bg_crops) >= int(target_count):
                    return bg_crops
    return bg_crops


def extract_resnet_features(crops, device, batch_size=64):
    """224x224 BGR kırpıntı listesinden dondurulmuş ResNet-50 ile L2-normalize 2048-D öznitelikler çıkarır."""
    if not crops:
        return np.empty((0, 2048), dtype=np.float32)

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    backbone.eval()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch_bgr = np.stack(crops[start : start + batch_size], axis=0)
            batch_rgb = batch_bgr[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()
            t_batch = torch.from_numpy(batch_rgb).float().to(device) / 255.0
            t_batch = (t_batch - im_mean) / im_std
            out = backbone(t_batch).flatten(1).cpu().numpy()
            feats.append(out.astype(np.float32))

    raw = np.vstack(feats)
    return l2_normalize(raw).astype(np.float32)


def fit_pca_and_oas_maha(U_tr, pca_var_ratio=0.90):
    """Eğitim öznitelikleri üzerinde ortalama, %90 varyans PCA alt-uzayı ve Chen et al. (2010)
    OAS büzülmeli kovaryans öz-ayrışımını hesaplar."""
    U64 = np.asarray(U_tr, dtype=np.float64)
    n_tr, d = U64.shape
    mu = np.mean(U64, axis=0, keepdims=True)
    Uc = U64 - mu

    _, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    eigvals_all = (S ** 2) / float(n_tr)
    total_var = float(np.sum(eigvals_all))
    if total_var <= 1e-15:
        raise ValueError("Eğitim öznitelik varyansı sıfıra çok yakın.")

    cum_var = np.cumsum(eigvals_all) / total_var
    p = int(np.searchsorted(cum_var, float(pca_var_ratio)) + 1)
    p = max(1, min(p, Vt.shape[0]))
    Vp = Vt[:p].astype(np.float32)

    # Oracle Approximating Shrinkage (OAS, Chen et al. 2010)
    tr_S = total_var
    tr_S2 = float(np.sum(eigvals_all ** 2))
    den = (float(n_tr) + 1.0 - 2.0 / float(d)) * (tr_S2 - (tr_S ** 2) / float(d))
    if den <= 1e-12:
        alpha = 1.0
    else:
        num = (1.0 - 2.0 / float(d)) * tr_S2 + (tr_S ** 2)
        alpha = float(np.clip(num / den, 0.0, 1.0))

    beta = alpha * (tr_S / float(d))
    nonzero_mask = S > 1e-12
    maha_V = Vt[nonzero_mask].astype(np.float32)
    maha_eigvals = eigvals_all[nonzero_mask].astype(np.float32)

    return (
        mu.astype(np.float32),
        Vp,
        maha_V,
        maha_eigvals,
        np.float32(alpha),
        np.float32(beta),
    )


def fit_l2_logistic_regression_irls(X, y, l2_reg=1.0, max_iter=50, tol=1e-10):
    """Saf NumPy Newton-IRLS kullanarak L2 düzenlileştirilmiş lojistik regresyon ağırlıklarını
    (weights, bias) deterministik olarak öğrenir."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n, d = X64.shape
    X_aug = np.hstack([X64, np.ones((n, 1), dtype=np.float64)])
    reg_diag = np.diag([float(l2_reg)] * d + [0.0])

    theta = np.zeros(d + 1, dtype=np.float64)
    for _ in range(int(max_iter)):
        z = X_aug @ theta
        p = sigmoid(z)
        grad = X_aug.T @ (p - y64) + reg_diag @ theta
        if np.linalg.norm(grad) < tol:
            break
        s = np.maximum(p * (1.0 - p), 1e-12)
        H = X_aug.T @ (s[:, None] * X_aug) + reg_diag
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.linalg.norm(step) < tol:
            break

    return theta[:d].astype(np.float64), float(theta[d])


def build_reference_bank(
    data_dir,
    out_path,
    pad_ratio=0.10,
    k=5,
    pca_var_ratio=0.90,
    l2_reg=1.0,
    target_bg=1800,
    seed=42,
    device=None,
):
    """Tüm veri kümesini okuyup sıfır sızıntılı referans bankasını oluşturur ve .npz olarak kaydeder."""
    data_root = Path(data_dir).expanduser().resolve()
    out_file = Path(out_path).expanduser().resolve()
    dev = select_device(device)

    print("=" * 78)
    print("TEMİZ RESNET-50 OOD REFERANS BANKASI OLUŞTURMA (SIFIR SIZINTI)")
    print("=" * 78)
    print(f"Veri Kümesi : {data_root}")
    print(f"Çıktı Dosya : {out_file}")
    print(f"Cihaz       : {dev} | pad_ratio={pad_ratio:.2f} | k={k} | pca_var={pca_var_ratio:.2f}")

    train_orig_paths = list_split_images(data_root / "images" / "train", exclude_synthetic=True)
    n_train_imgs, train_crops = collect_gt_crops(
        data_root, "train", pad_ratio=pad_ratio, exclude_synthetic=True
    )
    bg_crops = collect_background_crops(
        data_root, train_orig_paths, pad_ratio=pad_ratio, target_count=target_bg, seed=seed
    )
    n_val_imgs, val_crops = collect_gt_crops(data_root, "val", pad_ratio=pad_ratio)
    n_test_imgs, test_crops = collect_gt_crops(data_root, "test", pad_ratio=pad_ratio)

    print(f"Train (orijinal) : {n_train_imgs} görsel | {len(train_crops)} kutu")
    print(f"Arka Plan (IoU=0): {len(bg_crops)} yama")
    print(f"Val (kalibrasyon): {n_val_imgs} görsel | {len(val_crops)} kutu")
    print(f"Test (salt-okunur): {n_test_imgs} görsel | {len(test_crops)} kutu")

    U_tr = extract_resnet_features(train_crops, dev)
    U_bg = extract_resnet_features(bg_crops, dev)
    U_va = extract_resnet_features(val_crops, dev)
    U_te = extract_resnet_features(test_crops, dev)

    # Sadece U_tr üzerinde PCA + OAS Mahalanobis öğrenimi
    mu, Vp, maha_V, maha_eigvals, maha_alpha, maha_beta = fit_pca_and_oas_maha(
        U_tr, pca_var_ratio=pca_var_ratio
    )

    # Eğitim OOD sinyalleri (kendi kendine eşleşme yanlılığını önlemek için LOO)
    phi_tr = extract_ood_signals(
        U_tr,
        U_tr,
        U_bg,
        mu,
        Vp,
        maha_V,
        maha_eigvals,
        maha_alpha,
        maha_beta,
        k=k,
        exclude_self_tr=True,
    )
    phi_bg = extract_ood_signals(
        U_bg,
        U_tr,
        U_bg,
        mu,
        Vp,
        maha_V,
        maha_eigvals,
        maha_alpha,
        maha_beta,
        k=k,
        exclude_self_bg=True,
    )

    phi_train = np.vstack([phi_tr, phi_bg])
    y_train = np.concatenate([np.ones(len(phi_tr)), np.zeros(len(phi_bg))])
    feat_mean = np.mean(phi_train, axis=0)
    feat_std = np.maximum(np.std(phi_train, axis=0), 1e-6)
    X_train = (phi_train - feat_mean) / feat_std

    weights, bias = fit_l2_logistic_regression_irls(X_train, y_train, l2_reg=l2_reg)

    # Val ve Test sinyalleri (eğitim istatistikleriyle standartlaştırılır)
    phi_va = extract_ood_signals(
        U_va, U_tr, U_bg, mu, Vp, maha_V, maha_eigvals, maha_alpha, maha_beta, k=k
    )
    phi_te = extract_ood_signals(
        U_te, U_tr, U_bg, mu, Vp, maha_V, maha_eigvals, maha_alpha, maha_beta, k=k
    )

    z_tr = ((phi_tr - feat_mean) / feat_std) @ weights + bias
    z_bg = ((phi_bg - feat_mean) / feat_std) @ weights + bias
    z_va = ((phi_va - feat_mean) / feat_std) @ weights + bias
    z_te = ((phi_te - feat_mean) / feat_std) @ weights + bias

    # Yalnızca Val setinden TPR-95 kalibrasyonu
    tau_raw_95 = float(np.percentile(z_va, 5.0))
    score_scale = max(float(np.std(z_va)), 1e-3)

    s_tr = sigmoid((z_tr - tau_raw_95) / score_scale).astype(np.float32)
    s_bg = sigmoid((z_bg - tau_raw_95) / score_scale).astype(np.float32)
    s_va = sigmoid((z_va - tau_raw_95) / score_scale).astype(np.float32)
    s_te = sigmoid((z_te - tau_raw_95) / score_scale).astype(np.float32)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_file),
        U_tr=U_tr.astype(np.float32),
        U_bg=U_bg.astype(np.float32),
        mu=mu.astype(np.float32),
        Vp=Vp.astype(np.float32),
        maha_V=maha_V.astype(np.float32),
        maha_eigvals=maha_eigvals.astype(np.float32),
        maha_alpha=np.float32(maha_alpha),
        maha_beta=np.float32(maha_beta),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        weights=weights.astype(np.float32),
        bias=np.float32(bias),
        tau_raw_95=np.float32(tau_raw_95),
        score_scale=np.float32(score_scale),
        pad_ratio=np.float32(pad_ratio),
        pca_var_ratio=np.float32(pca_var_ratio),
        l2_reg=np.float32(l2_reg),
        k=np.int32(k),
        s_tr=s_tr,
        s_bg=s_bg,
        s_va=s_va,
        s_te=s_te,
        n_train_imgs=np.int32(n_train_imgs),
        n_train_boxes=np.int32(len(U_tr)),
        n_val_imgs=np.int32(n_val_imgs),
        n_val_boxes=np.int32(len(U_va)),
        n_test_imgs=np.int32(n_test_imgs),
        n_test_boxes=np.int32(len(U_te)),
    )

    size_mb = out_file.stat().st_size / (1024.0 * 1024.0)
    print("-" * 78)
    print(f"PCA Boyutu (p, >=%{pca_var_ratio*100:.0f} var) : {Vp.shape[0]} / 2048")
    print(f"OAS Shrinkage alpha              : {float(maha_alpha):.6f}")
    print(f"Öğrenilen Ağırlıklar (w)         : {[round(float(w), 4) for w in weights]}")
    print(f"Öğrenilen Bias (b)               : {bias:.4f}")
    print(f"Val TPR (@0.5000)                : {int(np.sum(s_va >= 0.5))}/{len(s_va)} (%{np.mean(s_va >= 0.5)*100:.2f})")
    print(f"Test TPR (@0.5000, salt-okunur)  : {int(np.sum(s_te >= 0.5))}/{len(s_te)} (%{np.mean(s_te >= 0.5)*100:.2f})")
    print(f"Arka Plan Reddetme (@0.5000)     : {int(np.sum(s_bg < 0.5))}/{len(s_bg)} (%{np.mean(s_bg < 0.5)*100:.2f})")
    print(f"Kaydedilen Banka Boyutu          : {size_mb:.2f} MB")
    print("=" * 78)


def main():
    """Komut satırı argümanlarını ayrıştırır ve referans bankası üretimini başlatır."""
    parser = argparse.ArgumentParser(
        description="Model A için sıfır sızıntılı saf ResNet-50 referans bankası oluşturucu"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT.parent / "v0.3_wild_boar_closeup"),
        help="YOLO veri kümesi kök klasörü",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "model" / "knn_bank.npz"),
        help="Çıktı .npz referans bankası dosya yolu",
    )
    parser.add_argument("--pad-ratio", type=float, default=0.10, help="Letterbox bağlam payı oranı")
    parser.add_argument("--k", type=int, default=5, help="KNN komşu sayısı")
    parser.add_argument("--pca-var-ratio", type=float, default=0.90, help="PCA açıklanan varyans oranı")
    parser.add_argument("--l2-reg", type=float, default=1.0, help="L2 lojistik regresyon katsayısı")
    parser.add_argument("--target-bg", type=int, default=1800, help="Hedef arka plan yama sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument("--device", type=str, default=None, help="Çıkarım cihazı (cuda/mps/cpu)")

    args = parser.parse_args()
    build_reference_bank(
        data_dir=args.data,
        out_path=args.out,
        pad_ratio=args.pad_ratio,
        k=args.k,
        pca_var_ratio=args.pca_var_ratio,
        l2_reg=args.l2_reg,
        target_bg=args.target_bg,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
