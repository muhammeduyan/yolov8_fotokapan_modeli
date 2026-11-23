#!/usr/bin/env python3
"""
generate_background_and_bank.py
===============================
1. Kaynak eğitim görsellerinin domuz kutusu DIŞINDA kalan (IoU = 0, güvenli mesafeli)
   taş, kaya, ağaç, çalı ve gece otları bölgelerinden:
   a) YOLO İnce Ayarı (Fine-Tuning) için 250 adet "Boş Arka Plan (Negative/Background)" görseli
      (ve boş .txt etiket dosyası) üretip hedef veri setinin `images/train` klasörüne ekler.
   b) KNN Filtresi için 1.800 adet "Arka Plan Referans Yaması (Background Reference Bank - R_bg)"
      çıkarır (özellikle yüksek dokulu kaya, taş, ağaç gövdesi ve çalı yamaları).
"""

import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


def read_gt_boxes(lbl_path, w, h, margin_ratio=0.08):
    """Gerçek domuz kutularını (ve etrafındaki güvenlik payını) piksel cinsinden döndürür."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        # Domuz kutusunun etrafına güvenlik payı bırak ki domuzun kılı bile arka plana girmesin
        bw_m = bw * (1.0 + 2 * margin_ratio)
        bh_m = bh * (1.0 + 2 * margin_ratio)
        x1 = max(0.0, (cx - bw_m / 2) * w)
        y1 = max(0.0, (cy - bh_m / 2) * h)
        x2 = min(float(w), (cx + bw_m / 2) * w)
        y2 = min(float(h), (cy + bh_m / 2) * h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def overlaps_any(crop, boxes):
    cx1, cy1, cx2, cy2 = crop
    for x1, y1, x2, y2 in boxes:
        if min(cx2, x2) > max(cx1, x1) and min(cy2, y2) > max(cy1, y1):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Domuz dışı bölgelerden YOLO arka plan görselleri ve KNN arka plan bankası üretir."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="../v0.3_wild_boar_closeup",
        help="Kaynak YOLO veri seti klasörü (images/train ve labels/train içeren)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="../v0.3_wild_boar_closeup",
        help="Üretilen boş arka plan görsellerinin ekleneceği hedef veri seti klasörü",
    )
    parser.add_argument(
        "--out-cache",
        type=str,
        default="outputs/v02_bg_patches.npz",
        help="KNN arka plan yamalarının kaydedileceği .npz dosyası",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()

    train_img_dir = src_dir / "images" / "train"
    train_lbl_dir = src_dir / "labels" / "train"

    out_img_dir = dst_dir / "images" / "train"
    out_lbl_dir = dst_dir / "labels" / "train"

    # Önce varsa eski *_bg*.jpg dosyalarını temizle
    for old_bg in out_img_dir.glob("*_bg*.jpg"):
        old_bg.unlink()
    for old_txt in out_lbl_dir.glob("*_bg*.txt"):
        old_txt.unlink()

    train_imgs = sorted(train_img_dir.glob("*.jpg"))
    rng = random.Random(42)

    target_yolo_bg_imgs = 250
    target_knn_bg_patches = 1800

    print("=" * 74)
    print("DOMUZ DIŞI BÖLGELERDEN ARKA PLAN GÖRSELİ & REFERANS BANKASI ÜRETİMİ")
    print("=" * 74)
    print(f"Kaynak Eğitim Görselleri : {len(train_imgs)} adet ({train_img_dir})")

    # İşlemi hızlandırmak için görsellerden domuzsuz bölgeleri paralel çıkaralım
    def process_one_image(img_path):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        im = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_COLOR_2)
        if im is None:
            return None
        h, w = im.shape[:2]
        # Üst ve alt fotokapan bilgi bantlarını hariç tut
        y_top = int(h * 0.06)
        y_bot = int(h * 0.91)
        gt_boxes = read_gt_boxes(lbl_path, w, h, margin_ratio=0.10)

        local_rng = random.Random(hash(img_path.name) & 0xFFFFFFFF)
        bg_scene = None
        bg_patches = []

        # 1. YOLO eğitimi için geniş arka plan penceresi (taş/kaya/ağaç/çalı sahnesi)
        for _ in range(25):
            cw = local_rng.randint(int(w * 0.28), int(w * 0.55))
            ch = int(cw * local_rng.uniform(0.70, 1.15))
            if ch >= (y_bot - y_top) or cw >= w:
                continue
            cx1 = local_rng.randint(0, w - cw)
            cy1 = local_rng.randint(y_top, y_bot - ch)
            crop_box = (cx1, cy1, cx1 + cw, cy1 + ch)
            if not overlaps_any(crop_box, gt_boxes):
                patch = im[cy1 : cy1 + ch, cx1 : cx1 + cw]
                # Çok karanlık/düz siyah olmayan, gerçek kaya/ağaç/ot dokusu içeren sahneleri seç
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                if gray.std() > 22.0 and cv2.Laplacian(gray, cv2.CV_64F).var() > 25.0:
                    bg_scene = cv2.resize(patch, (640, 640), interpolation=cv2.INTER_AREA)
                    break

        # 2. KNN Arka Plan Referans Bankası (R_bg) için domuz kutusu boyutlarında (kaya/ağaç/çalı) yamalar
        for _ in range(30):
            if len(bg_patches) >= 2:
                break
            pw = local_rng.randint(int(w * 0.08), int(w * 0.30))
            ph = int(pw * local_rng.uniform(0.65, 1.45))
            if ph >= (y_bot - y_top) or pw >= w:
                continue
            px1 = local_rng.randint(0, w - pw)
            py1 = local_rng.randint(y_top, y_bot - ph)
            p_box = (px1, py1, px1 + pw, py1 + ph)
            if not overlaps_any(p_box, gt_boxes):
                p_img = im[py1 : py1 + ph, px1 : px1 + pw]
                g = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
                # Yüksek kontrastlı / dokulu taş, kaya, ağaç gövdesi, çalı yamalarını önceliklendir
                if g.std() > 20.0 and cv2.Laplacian(g, cv2.CV_64F).var() > 20.0:
                    bg_patches.append((cv2.resize(p_img, (224, 224)), cv2.resize(p_img, (128, 128))))

        return (img_path.stem, bg_scene, bg_patches)

    scenes = []
    patches_224 = []
    patches_128 = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(process_one_image, train_imgs):
            if res is None:
                continue
            stem, scene, plist = res
            if scene is not None:
                scenes.append((stem, scene))
            for p224, p128 in plist:
                patches_224.append(p224)
                patches_128.append(p128)

    rng.shuffle(scenes)
    selected_scenes = scenes[:target_yolo_bg_imgs]

    for idx, (stem, scene_img) in enumerate(selected_scenes):
        name = f"{stem}_bg{idx:04d}"
        cv2.imwrite(str(out_img_dir / f"{name}.jpg"), scene_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # YOLO'da boş .txt dosyası = "Bu görselde hiç domuz yok, içindeki her şey arka plandır"
        (out_lbl_dir / f"{name}.txt").write_text("", encoding="utf-8")

    # KNN Arka Plan Referans yamalarını kaydet
    patches_224 = patches_224[:target_knn_bg_patches]
    patches_128 = patches_128[:target_knn_bg_patches]

    bg_cache_path = Path(args.out_cache)
    bg_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bg_cache_path,
        patches_224=np.stack(patches_224).astype(np.uint8),
        patches_128=np.stack(patches_128).astype(np.uint8),
    )

    total_train_imgs = len(list(out_img_dir.glob("*.jpg")))
    print(f"✅ YOLO Eğitimi İçin Eklenen Boş Arka Plan Görseli : {len(selected_scenes)} adet")
    print(f"✅ Yeni Toplam Train Görsel Sayısı                 : {total_train_imgs} (1200 orijinal + 600 close-up + {len(selected_scenes)} boş arka plan)")
    print(f"✅ KNN Arka Plan Bankası İçin Çıkarılan Kaya/Ağaç  : {len(patches_224)} adet yama ({bg_cache_path})")
    print("=" * 74)


if __name__ == "__main__":
    main()
