#!/usr/bin/env python3
"""
generate_closeup_crops.py
=========================

Mevcut YOLO veri setindeki yaban domuzu kutularından "yakın çekim" (close-up)
tarzı yeni eğitim görselleri üretir ve bunları orijinallerle birlikte YENİ bir
veri seti klasöründe birleştirir.

Ne yapar?
---------
1. Sadece `train` bölümündeki görselleri kullanır (val/test'e dokunmaz -> veri sızıntısı olmaz).
2. Her seçilen domuz kutusu için, domuz kırpılmış karenin %25-%85'ini kaplayacak
   şekilde rastgele bir pencere kırpar (= kamera domuza yaklaşmış gibi).
3. Bazı kırpmalarda domuzun bir kısmı kare dışında kalır (gerçek yakın çekimlerdeki gibi).
4. Kırpılan karede kalan TÜM domuz kutularının etiketlerini yeniden hesaplar.
   Bir domuz yarım yamalak (çok az) görünüyorsa o kırpmayı atar ve yeniden dener
   (etiketsiz domuz parçası modele "bu arka plan" diye yanlış öğretmesin diye).
5. Orijinal veri setine ASLA yazmaz. Yeni klasörde:
     images/train  = 1.200 orijinal (hardlink, ek disk alanı yok) + üretilen close-up'lar
     images/val    = 300 orijinal (hardlink, değişmeden)
     images/test   = 300 orijinal (hardlink, değişmeden)
   ve hazır bir `data.yaml` oluşturur.

Kullanım
--------
    python scripts/generate_closeup_crops.py \
        --src ../v0.2_wild_boar \
        --dst ../v0.3_wild_boar_closeup \
        --ratio 0.5 --preview 24

Önce sadece kontrol etmek için (hiçbir şey yazmadan sayıları görmek):
    python scripts/generate_closeup_crops.py --src ... --dst ... --dry-run
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Yardımcı fonksiyonlar
# --------------------------------------------------------------------------- #
def read_labels(label_path, img_w, img_h):
    """YOLO etiketini (cls cx cy w h, normalize) piksel xyxy listesine çevirir."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls, max(0.0, x1), max(0.0, y1), min(img_w, x2), min(img_h, y2)))
    return boxes


def link_or_copy(src, dst):
    """Hardlink dener (disk alanı harcamaz, orijinal dosya hiç değişmez); olmazsa kopyalar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_size_fast(path):
    """Görseli tamamen açmadan boyutunu bulmaya çalışır; olmazsa açar."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return img.shape[1], img.shape[0]


def propose_crop(box, img_w, img_h, target_cover, allow_truncate, rng, y_min=0.0, y_max=None):
    if y_max is None:
        y_max = img_h
    """
    Hedef kutunun kırpılmış karenin `target_cover` kadarını kaplayacağı bir pencere önerir.
    Dönen değer: (cx1, cy1, cx2, cy2) piksel cinsinden veya None.
    """
    _, x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    # Kırpma en-boy oranı: yatay 4:3 ile dikey 3:4 arası rastgele
    aspect = rng.uniform(0.75, 1.3333)  # w / h

    crop_area = (bw * bh) / target_cover
    ch = np.sqrt(crop_area / aspect)
    cw = ch * aspect

    if not allow_truncate:
        # Domuzun tamamı karede kalsın
        cw = max(cw, bw * 1.05)
        ch = max(ch, bh * 1.05)

    # Görselden (kullanılabilir dikey bölgeden) büyük olamaz
    usable_h = y_max - y_min
    if cw > img_w or ch > usable_h:
        s = min(img_w / cw, usable_h / ch)
        cw, ch = cw * s, ch * s

    if allow_truncate:
        # Domuzun %60-%100'ü görünecek şekilde konumla
        min_vis = rng.uniform(0.6, 0.9)
        lo_x = x1 + bw * min_vis - cw  # pencere en sola kayabilir
        hi_x = x2 - bw * min_vis
        lo_y = y1 + bh * min_vis - ch
        hi_y = y2 - bh * min_vis
    else:
        lo_x, hi_x = x2 - cw, x1
        lo_y, hi_y = y2 - ch, y1

    lo_x, hi_x = max(0.0, lo_x), min(img_w - cw, hi_x)
    lo_y, hi_y = max(y_min, lo_y), min(y_max - ch, hi_y)
    if lo_x > hi_x or lo_y > hi_y:
        return None

    cx1 = rng.uniform(lo_x, hi_x)
    cy1 = rng.uniform(lo_y, hi_y)
    return cx1, cy1, cx1 + cw, cy1 + ch


def remap_boxes(boxes, crop, min_keep_vis, max_partial_vis, min_px):
    """
    Kırpma penceresi içindeki kutuları yeniden hesaplar.
    - Görünürlüğü >= min_keep_vis olanlar etiketlenir.
    - 0 < görünürlük < max_partial_vis olan (yarım yamalak) bir domuz varsa None döner
      (kırpma reddedilir), çünkü etiketsiz domuz parçası modele yanlış öğretir.
    """
    cx1, cy1, cx2, cy2 = crop
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        ix1, iy1 = max(x1, cx1), max(y1, cy1)
        ix2, iy2 = min(x2, cx2), min(y2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue  # tamamen dışarıda
        vis = (ix2 - ix1) * (iy2 - iy1) / area
        if vis >= min_keep_vis and (ix2 - ix1) >= min_px and (iy2 - iy1) >= min_px:
            out.append((cls, ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1))
        else:
            return None  # yarım yamalak görünen domuz -> bu kırpmayı reddet
    return out if out else None


def to_yolo_lines(boxes, w, h):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def draw_preview(img, boxes):
    vis = img.copy()
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    return vis


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Yaban domuzu close-up veri artırma")
    ap.add_argument("--src", required=True, help="Orijinal veri seti (images/, labels/ içeren)")
    ap.add_argument("--dst", required=True, help="Yeni birleşik veri setinin yazılacağı BOŞ klasör")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="Üretilecek close-up sayısı / orijinal train görsel sayısı (varsayılan 0.5 -> ~600)")
    ap.add_argument("--min-cover", type=float, default=0.25, help="Domuzun karede kapladığı min oran")
    ap.add_argument("--max-cover", type=float, default=0.85, help="Domuzun karede kapladığı max oran")
    ap.add_argument("--truncate-prob", type=float, default=0.3,
                    help="Domuzun bir kısmının kare dışında kalacağı kırpma olasılığı")
    ap.add_argument("--min-box-px", type=int, default=160,
                    help="Orijinalde kısa kenarı bundan küçük kutular atlanır (bulanık büyütme olmasın)")
    ap.add_argument("--out-max-side", type=int, default=1280, help="Üretilen görselin uzun kenarı (px)")
    ap.add_argument("--flip-prob", type=float, default=0.5, help="Yatay çevirme olasılığı")
    ap.add_argument("--tries", type=int, default=15, help="Kutu başına geçerli kırpma deneme sayısı")
    ap.add_argument("--exclude-top", type=float, default=0.06,
                    help="Görselin üstünden kırpmaya dahil edilmeyecek oran (fotokapan bilgi bandı)")
    ap.add_argument("--exclude-bottom", type=float, default=0.09,
                    help="Görselin altından kırpmaya dahil edilmeyecek oran (tarih/saat bandı)")
    ap.add_argument("--min-sharpness", type=float, default=15.0,
                    help="Laplacian varyansı bundan düşük (bulanık/gürültülü) kırpmalar atlanır. 0 = kapalı")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", type=int, default=0, help="Kontrol için kaç örnek önizleme kaydedilsin")
    ap.add_argument("--dry-run", action="store_true", help="Hiçbir şey yazmadan sadece istatistik göster")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    # --- Güvenlik kontrolleri: orijinal veri asla değişmesin ---
    if not (src / "images" / "train").is_dir() or not (src / "labels" / "train").is_dir():
        sys.exit(f"HATA: {src} içinde images/train ve labels/train bulunamadı.")
    if dst == src or src in dst.parents or dst in src.parents:
        sys.exit("HATA: --dst, --src ile aynı olamaz ve iç içe olamaz (orijinal veri korunmalı).")
    if dst.exists() and any(dst.iterdir()) and not args.dry_run:
        sys.exit(f"HATA: {dst} boş değil. Lütfen yeni/boş bir klasör adı verin.")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    train_imgs = sorted(p for p in (src / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXTS)
    n_target = int(round(len(train_imgs) * args.ratio))

    # --- 1) Uygun (yeterince büyük) kutuları topla ---
    candidates = []  # (img_path, label_path, (w,h), box_index)
    all_boxes = {}
    for ip in train_imgs:
        lp = src / "labels" / "train" / (ip.stem + ".txt")
        size = image_size_fast(ip)
        if size is None:
            continue
        w, h = size
        boxes = read_labels(lp, w, h)
        all_boxes[ip] = (lp, (w, h), boxes)
        for i, (_, x1, y1, x2, y2) in enumerate(boxes):
            if min(x2 - x1, y2 - y1) >= args.min_box_px:
                candidates.append((ip, i))

    print("=" * 70)
    print("CLOSE-UP VERİ ARTIRMA")
    print("=" * 70)
    print(f"Kaynak            : {src}")
    print(f"Hedef             : {dst}")
    print(f"Orijinal train    : {len(train_imgs)} görsel, {sum(len(v[2]) for v in all_boxes.values())} kutu")
    print(f"Uygun kutu (>= {args.min_box_px}px): {len(candidates)}")
    print(f"Üretilecek hedef  : {n_target} close-up (ratio={args.ratio})")
    if not candidates:
        sys.exit("HATA: Uygun kutu yok. --min-box-px değerini düşürün.")
    if args.dry_run:
        print("\n[--dry-run] Hiçbir dosya yazılmadı.")
        return

    # --- 2) Orijinalleri yeni klasöre bağla (hardlink, ek alan yok) ---
    print("\n[1/3] Orijinal görseller ve etiketler bağlanıyor (train/val/test)...")
    for split in ("train", "val", "test"):
        si, sl = src / "images" / split, src / "labels" / split
        if not si.is_dir():
            continue
        for p in si.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                link_or_copy(p, dst / "images" / split / p.name)
        if sl.is_dir():
            for p in sl.glob("*.txt"):
                link_or_copy(p, dst / "labels" / split / p.name)

    # --- 3) Close-up üret ---
    print(f"[2/3] {n_target} close-up üretiliyor...")
    out_img_dir = dst / "images" / "train"
    out_lbl_dir = dst / "labels" / "train"
    prev_dir = dst / "_preview"
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    # Kutuları karıştır; hedef sayıya ulaşana kadar (gerekirse tekrar ederek) dolaş
    order = candidates[:]
    rng.shuffle(order)
    manifest = []
    made, attempts_pass, idx = 0, 0, 0
    rejected_blur = 0
    cache = {"path": None, "img": None}

    while made < n_target:
        if idx >= len(order):
            idx = 0
            attempts_pass += 1
            rng.shuffle(order)
            if attempts_pass > 5:
                print("Uyarı: Yeterli geçerli kırpma bulunamadı, erken durduruldu.")
                break
        ip, bi = order[idx]
        idx += 1
        lp, (w, h), boxes = all_boxes[ip]

        crop, new_boxes, cover, trunc = None, None, None, False
        for _ in range(args.tries):
            cover = rng.uniform(args.min_cover, args.max_cover)
            trunc = rng.random() < args.truncate_prob
            c = propose_crop(boxes[bi], w, h, cover, trunc, rng,
                             y_min=h * args.exclude_top, y_max=h * (1.0 - args.exclude_bottom))
            if c is None:
                continue
            nb = remap_boxes(boxes, c, min_keep_vis=0.5, max_partial_vis=0.5, min_px=8)
            if nb is not None:
                crop, new_boxes = c, nb
                break
        if crop is None:
            continue

        if cache["path"] != ip:
            cache["img"] = cv2.imread(str(ip))
            cache["path"] = ip
        img = cache["img"]
        if img is None:
            continue
        H, W = img.shape[:2]
        if (W, H) != (w, h):  # EXIF vb. farkı varsa ölçekle
            sx, sy = W / w, H / h
            crop = (crop[0] * sx, crop[1] * sy, crop[2] * sx, crop[3] * sy)
            new_boxes = [(c_, x1 * sx, y1 * sy, x2 * sx, y2 * sy) for c_, x1, y1, x2, y2 in new_boxes]

        cx1, cy1, cx2, cy2 = map(int, map(round, crop))
        patch = img[cy1:cy2, cx1:cx2]
        if patch.size == 0:
            continue
        ph, pw = patch.shape[:2]

        # Netlik filtresi: büyütülünce gürültüye dönüşen bulanık kırpmaları at
        if args.min_sharpness > 0:
            g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (640, max(1, int(640 * ph / pw))), interpolation=cv2.INTER_AREA)
            if cv2.Laplacian(g, cv2.CV_64F).var() < args.min_sharpness:
                rejected_blur += 1
                continue
        # Yeniden boyutlandır (uzun kenar out_max_side)
        s = min(args.out_max_side / max(ph, pw), 2.0)  # küçült ya da en fazla 2x büyüt
        out_w, out_h = int(round(pw * s)), int(round(ph * s))
        patch = cv2.resize(patch, (out_w, out_h),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        new_boxes = [(c_, x1 * s, y1 * s, x2 * s, y2 * s) for c_, x1, y1, x2, y2 in new_boxes]

        flipped = rng.random() < args.flip_prob
        if flipped:
            patch = cv2.flip(patch, 1)
            new_boxes = [(c_, out_w - x2, y1, out_w - x1, y2) for c_, x1, y1, x2, y2 in new_boxes]

        name = f"{ip.stem}_closeup{made:04d}"
        cv2.imwrite(str(out_img_dir / f"{name}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl_dir / f"{name}.txt").write_text("\n".join(to_yolo_lines(new_boxes, out_w, out_h)) + "\n")

        if made < args.preview:
            cv2.imwrite(str(prev_dir / f"{name}.jpg"), draw_preview(patch, new_boxes))

        manifest.append({
            "closeup": f"{name}.jpg", "source": ip.name,
            "crop_x1": cx1, "crop_y1": cy1, "crop_x2": cx2, "crop_y2": cy2,
            "target_cover": round(cover, 3), "truncated": int(trunc), "flipped": int(flipped),
            "n_boxes": len(new_boxes),
        })
        made += 1
        if made % 100 == 0:
            print(f"   {made}/{n_target}")

    # --- 4) Manifest ve data.yaml ---
    print("[3/3] manifest ve data.yaml yazılıyor...")
    with open(dst / "closeup_manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["closeup"])
        wr.writeheader()
        wr.writerows(manifest)

    (dst / "data.yaml").write_text(
        f"path: {dst}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: wild_boar\n",
        encoding="utf-8",
    )

    n_train_total = len([p for p in out_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print("=" * 70)
    print(f"Üretilen close-up : {made}")
    print(f"Bulanık diye atlanan kırpma: {rejected_blur}")
    print(f"Yeni train toplamı: {n_train_total}  (= {len(train_imgs)} orijinal + {made} close-up)")
    print(f"val / test        : değiştirilmedi (orijinal 300 / 300)")
    print(f"data.yaml         : {dst / 'data.yaml'}")
    print(f"Manifest          : {dst / 'closeup_manifest.csv'}")
    if args.preview:
        print(f"Önizleme          : {prev_dir}  (yeşil kutular etiketlerdir, kontrol edin!)")
    print(f"Orijinal veri seti: {src}  (HİÇ DEĞİŞTİRİLMEDİ)")


if __name__ == "__main__":
    main()
