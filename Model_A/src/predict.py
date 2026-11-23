#!/usr/bin/env python3
"""
predict.py: Model A (Local Wild Boar v0.4) için Tespit ve Kalibreli KNN/OOD Filtreleme CLI Aracı.

Kullanım:
    python src/predict.py --calibrate
    python src/predict.py --source samples/02_same_goat.jpg
    python src/predict.py --benchmark
"""

import argparse
from pathlib import Path
import pandas as pd

try:
    from src.detector import ModelAKNNFilter, PROJECT_ROOT
except ImportError:
    from detector import ModelAKNNFilter, PROJECT_ROOT


def _resolve_source_path(source_path):
    """Göreli görsel yolunu önce çalışma dizininde, bulunamazsa Model_A kökünde arar."""
    p = Path(str(source_path)).expanduser()
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists():
            return str(candidate)
    return str(source_path)


def print_calibration_report(detector):
    """Referans bankası istatistiklerini ve otomatik kalibrasyon (tau) özetini yazdırır."""
    stats = detector.get_calibration_summary()
    n_bg_patches = stats.get("n_bg_patches", 0)
    print("\n" + "=" * 78)
    print("OTOMATİK EŞİK KALİBRASYONU RAPORU (PERCENTILE / TPR-95 KALİBRASYONU)")
    print("=" * 78)
    print(f"Model / Referans Bankası  : {Path(stats['model_path']).name} + {Path(stats['bank_path']).name}")
    print(f"1. Domuz Referans Bankası : {stats['n_train_imgs']} görsel | {stats['n_train_boxes']} gerçek yaban domuzu kutusu")
    print(f"2. Arka Plan Bankası (BG) : {n_bg_patches} adet taş/kaya/ağaç/çalı/ot yaması (IoU=0)")
    print(f"3. Kalibrasyon Seti (Val) : {stats['n_val_imgs']} görsel  | {stats['n_val_boxes']} gerçek yaban domuzu kutusu")
    print(f"4. Doğrulama Seti (Test)  : {stats['n_test_imgs']} görsel  | {stats['n_test_boxes']} gerçek yaban domuzu kutusu")
    print("-" * 78)
    print(f"DOĞRULAMA (VAL) SETİ BENZERLİK DAĞILIMI ({stats['n_val_boxes']} Gerçek Yaban Domuzu):")
    print(f"   • Ortalama (Mean) Benzerlik    : {stats['val_mean']:.4f}")
    print(f"   • Ortanca (Median / %50)       : {stats['val_median']:.4f}")
    print(f"   • 10. Yüzdelik Dilim (%90 TPR) : {stats['val_p10']:.4f}")
    print(f"   •  5. Yüzdelik Dilim (%95 TPR) : {stats['val_p05']:.4f}  <-- [OTOMATİK SEÇİLEN EŞİK (tau)]")
    print(f"   •  1. Yüzdelik Dilim (%99 TPR) : {stats['val_p01']:.4f}")
    print("-" * 78)
    print(f"✅ OTOMATİK KALİBRE EDİLEN EŞİK (tau) : {stats['calibrated_tau']:.4f}")
    print(
        f"✅ VAL SETİ GERÇEK DOMUZ KORUMA (TPR) : {stats['val_tpr_count']}/{stats['n_val_boxes']} "
        f"(%{stats['val_tpr_pct']:.2f})"
    )
    print(
        f"✅ TEST SETİ GERÇEK DOMUZ KORUMA (TPR): {stats['test_tpr_count']}/{stats['n_test_boxes']} "
        f"(%{stats['test_tpr_pct']:.2f}) | Test Ortanca Benzerlik: {stats['test_median']:.4f}"
    )
    if "bg_reject_pct" in stats:
        print(f"🪨 KAYA/AĞAÇ ARKA PLAN REDDETME ORANI : %{stats['bg_reject_pct']:.2f} ({n_bg_patches} arka plan yamasında)")
    print("=" * 78)


def run_single_prediction(detector, source_path, save_dir, conf=0.25):
    """Tek bir görselde ham YOLO ve KNN/OOD filtreli tespitleri çalıştırıp sonuçları raporlar."""
    resolved_source = _resolve_source_path(source_path)
    stats = detector.get_calibration_summary()
    n_boar_boxes = f"{stats['n_train_boxes']:,}".replace(",", ".")
    n_bg_patches = f"{stats.get('n_bg_patches', 0):,}".replace(",", ".")
    target_tpr = 100.0 - float(getattr(detector, "percentile", 5.0))

    mode_label = (
        f"OTOMATİK KALİBRELİ (%{target_tpr:g} Val TPR -> tau={detector.threshold:.4f})"
        if detector.is_auto_calibrated
        else f"MANUEL ({detector.threshold:.4f})"
    )
    print("=" * 78)
    print(f"GÖRSEL ANALİZİ : {resolved_source}")
    print(f"Eşik Değeri    : {mode_label}")
    print(f"Çift Referans  : {n_boar_boxes} Domuz Kutusu + {n_bg_patches} Kaya/Ağaç/Çalı Arka Plan Yaması (IoU=0)")
    print("=" * 78)

    res = detector.predict(resolved_source, conf=conf, save_dir=save_dir)

    print(f"\n1. Standart Model A (Filtresiz):")
    if res["raw_detections"]:
        for i, d in enumerate(res["raw_detections"], 1):
            print(f"   🔸 Kutu {i}: {d['class']} | YOLO Güven Skoru: %{d['conf']*100:.1f} ({d['conf']:.4f})")
    else:
        print("   Tespit yok (Boş).")

    print(f"\n2. Otomatik Kalibreli KNN / Alt-Uzay Filtreli Model A:")
    if res["filtered_detections"]:
        for d in res["filtered_detections"]:
            print(
                f"   ✅ ONAYLANDI (GERÇEK YABAN DOMUZU): {d['class']} | "
                f"YOLO Güven: %{d['conf']*100:.1f} | Domuz Benzerliği: {d['sim']:.4f} (>= Eşik {detector.threshold:.4f})"
            )
    if res["suppressed_detections"]:
        for d in res["suppressed_detections"]:
            print(
                f"   🚫 ELENDİ (KUTU SİLİNDİ)          : "
                f"YOLO Güven: %{d['conf']*100:.1f} | Domuz Benzerliği: {d['sim']:.4f} (< Eşik {detector.threshold:.4f})"
            )
    if not res["filtered_detections"] and not res["suppressed_detections"]:
        print("   Tespit yok (Boş).")

    if res["saved_files"]:
        print(f"\nKaydedilen Görseller:")
        print(f"   Ham Model A Çıktısı   : {res['saved_files']['raw']}")
        print(f"   Filtreli Model Çıktısı: {res['saved_files']['filtered']}")
    print("=" * 78)


def run_benchmark(detector):
    """Kalibrasyon özetini ve negatif OOD challenge görselleri üzerindeki yanlış pozitif oranlarını raporlar."""
    print_calibration_report(detector)

    manifest_path = PROJECT_ROOT / "model" / "benchmark_manifest.csv"
    if not manifest_path.exists():
        print(f"Hata: Manifest bulunamadı: {manifest_path}")
        return

    df = pd.read_csv(str(manifest_path), encoding="utf-8-sig")
    if df.empty or not {"expected_class", "filename"}.issubset(df.columns):
        print(f"Hata: Geçersiz veya boş benchmark manifesti: {manifest_path}")
        return

    d = detector.calibration_data
    if d is None or "s_ch" not in d or "chal_fnames" not in d:
        print("Hata: Referans bankasında challenge benchmark verileri (s_ch, chal_fnames) bulunamadı.")
        return

    s_ch = d["s_ch"]
    chal_fnames = d["chal_fnames"]
    tau = detector.threshold

    print("\n" + "=" * 78)
    print(f"{len(df)} NEGATİF CHALLENGE GÖRSELİ BENCHMARK TESTİ (Otomatik Eşik tau = {tau:.4f})")
    print("=" * 78)

    results_by_species = {}
    for species, group in df.groupby("expected_class"):
        total = len(group)
        fnames_set = set(group["filename"].tolist())
        raw_fp_files = set()
        filt_fp_files = set()

        for fn, score in zip(chal_fnames, s_ch):
            if fn in fnames_set:
                raw_fp_files.add(fn)
                if score >= tau:
                    filt_fp_files.add(fn)

        results_by_species[species] = (len(raw_fp_files), len(filt_fp_files), total)

    print(f"\n{'Tür (Species)':<22} | {'Ham Model A (FP)':<22} | {'Kalibreli KNN (FP)':<22}")
    print("-" * 74)

    tot_raw, tot_filt, tot_all = 0, 0, 0
    for sp, (raw, filt, tot) in sorted(results_by_species.items()):
        raw_pct = (raw / tot * 100) if tot > 0 else 0.0
        filt_pct = (filt / tot * 100) if tot > 0 else 0.0
        print(
            f"{sp:<22} | {raw:2d}/{tot} (%{raw_pct:5.1f})          | {filt:2d}/{tot} (%{filt_pct:5.1f})"
        )
        tot_raw += raw
        tot_filt += filt
        tot_all += tot

    print("-" * 74)
    reduction = ((tot_raw - tot_filt) / tot_raw * 100) if tot_raw > 0 else 0.0
    tot_raw_pct = (tot_raw / tot_all * 100) if tot_all > 0 else 0.0
    tot_filt_pct = (tot_filt / tot_all * 100) if tot_all > 0 else 0.0
    print(
        f"{'GENEL TOPLAM':<22} | {tot_raw:2d}/{tot_all} (%{tot_raw_pct:5.1f})         | {tot_filt:2d}/{tot_all} (%{tot_filt_pct:5.1f})"
    )
    print(f"\n🎯 Yanlış Pozitif (False Positive) Azaltma Başarısı : %{reduction:.1f} ({tot_raw} -> {tot_filt})")
    stats = detector.get_calibration_summary()
    print(
        f"🐗 Gerçek Yaban Domuzu Koruma Oranı (Test ID TPR)  : %{stats['test_tpr_pct']:.2f} "
        f"({stats['test_tpr_count']}/{stats['n_test_boxes']} kutu)"
    )
    print("=" * 78)


def main():
    """Komut satırı argümanlarını ayrıştırır ve kalibrasyon, benchmark veya tekil görsel tahmini çalıştırır."""
    parser = argparse.ArgumentParser(
        description="Model A Otomatik Kalibreli KNN/Alt-Uzay Tabanlı OOD Filtreleme"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="samples/02_same_goat.jpg",
        help="Test edilecek görsel yolu",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Benzerlik eşik değeri (Boş bırakılırsa Val setinden otomatik kalibre edilen tau kullanılır)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=5.0,
        help="Doğrulama (val) seti yüzdelik dilimi (varsayılan: 5.0 -> %%95 TPR garantisi)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO minimum güven eşiği (varsayılan: 0.25)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "knn_predictions"),
        help="Sonuç görsellerinin kaydedileceği klasör (varsayılan: Model_A/outputs/knn_predictions)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Kalibre edilmiş referans bankası üzerinden otomatik eşik kalibrasyonu raporunu göster",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Hem gerçek domuz (Val/Test) koruma oranını hem de 100 negatif OOD challenge testini çalıştır",
    )

    args = parser.parse_args()

    if not (0.0 <= args.percentile <= 100.0):
        parser.error("--percentile değeri 0 ile 100 arasında olmalıdır.")
    if not (0.0 <= args.conf <= 1.0):
        parser.error("--conf değeri 0.0 ile 1.0 arasında olmalıdır.")

    detector = ModelAKNNFilter(threshold=args.threshold, percentile=args.percentile)

    if args.calibrate:
        print_calibration_report(detector)
    elif args.benchmark:
        run_benchmark(detector)
    else:
        run_single_prediction(detector, args.source, args.save_dir, conf=args.conf)


if __name__ == "__main__":
    main()
