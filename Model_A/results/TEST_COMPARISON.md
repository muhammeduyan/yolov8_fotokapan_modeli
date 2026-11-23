# Model A Test Karşılaştırma Raporu (Ham YOLOv8n vs. KNN/OOD Filtreli)

Bu rapor, **Ham Model A (KNN/OOD olmadan YOLOv8n)** ile **KNN/OOD Filtreli Model A (YOLOv8n + ResNet-50 Çift Bankalı KNN & Alt-Uzay Filtresi)** modellerinin aynı **300 görsellik test seti** üzerindeki karşılaştırmalı sonuçlarını içerir.

---

## 1. Test Kurulumu

| Parametre | Değer |
|---|---|
| **Test Veri Seti** | `v0.3_wild_boar_closeup/images/test` (`300` görsel, **`443`** `wild_boar` kutusu) |
| **Dedektör Modeli** | `Model_A/model/best.pt` (YOLOv8n) |
| **KNN / OOD Referans Bankası** | `Model_A/model/knn_bank.npz` (`1.811` domuz kutusu, `1.738` arka plan yaması) |
| **Kalibrasyon Eşiği (`tau`)** | `0.7000` (otomatik TPR-95 eşiği, `percentile = 5.0`) |
| **Çalışma Noktası / Standart mAP** | Kutu metrikleri `conf >= 0.25`; standart `mAP@0.5` ve `mAP@0.5:0.95` ise `conf = 0.001` |
| **Donanım / Cihaz** | Apple Silicon **`mps`** |

---

## 2. Ana Karşılaştırma Tablosu (Test Seti — 300 Görsel, 443 GT Kutusu)

| Metrik | Ham Model A (KNN/OOD Yok) | KNN/OOD Filtreli Model A (`tau=0.7000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `442` | `419` | **`-23`** (`-%5.20`) |
| **True Positives (TP, `IoU >= 0.50`)** | `384` | `376` | `-8` |
| **False Positives (FP, `IoU >= 0.50`)** | `58` | `43` | **`-15`** (`-%25.86`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `59` | `67` | `+8` |
| **Precision (`IoU >= 0.50`)** | `0.868778` (`%86.88`) | `0.897375` (`%89.74`) | **`+0.028597`** (`+2.86` puan) |
| **Recall (`IoU >= 0.50`)** | `0.866817` (`%86.68`) | `0.848758` (`%84.88`) | `-0.018059` (`-1.81` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.867797` (`%86.78`) | `0.872390` (`%87.24`) | **`+0.004593`** (`+0.46` puan) |
| **Precision (`IoU >= 0.75`)** | `0.807692` (`%80.77`) | `0.840095` (`%84.01`) | **`+0.032403`** (`+3.24` puan) |
| **Recall (`IoU >= 0.75`)** | `0.805869` (`%80.59`) | `0.794582` (`%79.46`) | `-0.011287` (`-1.13` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.806780` (`%80.68`) | `0.816705` (`%81.67`) | **`+0.009925`** (`+0.99` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.860887` | `0.843718` | `-0.017169` |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.921035` | `0.895723` | `-0.025312` |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.749471` | `0.734067` | `-0.015404` |

> *Not: Tek sınıflı (`wild_boar`) modelde `mAP@0.5` ile `AP@50` aynı metriktir. Ham Model A için Ultralytics `yolo val (split=test)` çapraz doğrulaması `mAP@0.5 = 0.923150`, `mAP@0.5:0.95 = 0.745972` olarak doğrulanmıştır.*

---

## 3. Kısa Değerlendirme

- **Yanlış Pozitif Azalması ve F1 Artışı:** KNN/OOD filtresi, 300 test görselindeki yanlış pozitifleri `58` → `43` (`-%25.86`) düşürerek Precision'ı `+2.86` puan (`%86.88` → `%89.74`), `F1-Score`'u ise `IoU >= 0.50` seviyesinde `+0.46` puan (`%86.78` → `%87.24`) ve `IoU >= 0.75` seviyesinde `+0.99` puan (`%80.68` → `%81.67`) artırmıştır.
- **Recall ve mAP Ödünleşimi:** Eşik altında kalan `8` gerçek domuz kutusu nedeniyle Recall `-1.81` puan (`%86.68` → `%84.88`), standart `mAP@0.5 (conf = 0.001)` ise `0.9210` → `0.8957` (`-0.0253`) olarak gerçekleşmiştir.
