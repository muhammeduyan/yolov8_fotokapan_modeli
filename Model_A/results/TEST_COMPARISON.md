# Model A Test Karşılaştırma Raporu (Ham YOLOv8n vs. KNN/OOD Filtreli)

Bu rapor, **Ham Model A (KNN/OOD olmadan YOLOv8n)** ile **KNN/OOD Filtreli Model A (YOLOv8n + ResNet-50 Çift Bankalı KNN & Alt-Uzay Filtresi)** modellerinin hem **300 görsellik ID test seti** (`v0.3_wild_boar_closeup/images/test`) hem de **378 görsellik karma test seti** (`/Users/muhammed/Desktop/test`) üzerindeki karşılaştırmalı sonuçlarını içerir.

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

## 3. Ana Karşılaştırma Tablosu (Karma Test Seti — 378 Görsel, 130 GT `wild_boar` Kutusu)

> Test seti: `/Users/muhammed/Desktop/test` (`80` `wild_boar` + `80` `wild_goat` + `79` `equus_quagga` + `79` `loxodonta_africana` + `60` `empty`; toplam `130` adet GT `wild_boar` kutusu, yalnızca `wild_boar` sınıfında GT bulunur). Çalışma noktası `conf >= 0.25`, standart mAP `conf = 0.001`, kalibrasyon eşiği `tau = 0.7000`, cihaz `mps`.

| Metrik | Ham Model A (KNN/OOD Yok) | KNN/OOD Filtreli Model A (`tau=0.7000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `313` | `139` | **`-174`** (`-%55.59`) |
| **True Positives (TP, `IoU >= 0.50`)** | `108` | `106` | `-2` |
| **False Positives (FP, `IoU >= 0.50`)** | `205` | `33` | **`-172`** (`-%83.90`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `22` | `24` | `+2` |
| **Precision (`IoU >= 0.50`)** | `0.345048` (`%34.50`) | `0.762590` (`%76.26`) | **`+0.417542`** (`+41.75` puan) |
| **Recall (`IoU >= 0.50`)** | `0.830769` (`%83.08`) | `0.815385` (`%81.54`) | `-0.015384` (`-1.54` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.487585` (`%48.76`) | `0.788104` (`%78.81`) | **`+0.300519`** (`+30.05` puan) |
| **Precision (`IoU >= 0.75`)** | `0.309904` (`%30.99`) | `0.690647` (`%69.06`) | **`+0.380743`** (`+38.07` puan) |
| **Recall (`IoU >= 0.75`)** | `0.746154` (`%74.62`) | `0.738462` (`%73.85`) | `-0.007692` (`-0.77` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.437923` (`%43.79`) | `0.713755` (`%71.38`) | **`+0.275832`** (`+27.58` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.622852` | `0.790152` | **`+0.167300`** |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.641030` | `0.829782` | **`+0.188752`** |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.534643` | `0.678458` | **`+0.143815`** |

### Kategori Bazlı Dağılım (378 Görsel, `conf >= 0.25`)

| Kategori | Görsel Sayısı | Ham Model A (Tespitli Görsel / FP) | KNN/OOD Filtreli (Tespitli Görsel / FP) | FP Azalması |
|---|---:|---:|---:|---:|
| `wild_boar` | `80` | `80` (`%100.00`) / `18` (`108` TP) | `79` (`%98.75`) / `11` (`106` TP) | `-7` (`-%38.89`) |
| `wild_goat` | `80` | `66` (`%82.50`) / `111` | `12` (`%15.00`) / `14` | **`-97`** (`-%87.39`) |
| `equus_quagga` | `79` | `21` (`%26.58`) / `31` | `4` (`%5.06`) / `4` | **`-27`** (`-%87.10`) |
| `loxodonta_africana` | `79` | `28` (`%35.44`) / `33` | `2` (`%2.53`) / `2` | **`-31`** (`-%93.94`) |
| `empty` | `60` | `8` (`%13.33`) / `12` | `2` (`%3.33`) / `2` | **`-10`** (`-%83.33`) |
| **Toplam** | **`378`** | `203` / `205` (`108` TP) | `99` / `33` (`106` TP) | **`-172`** (`-%83.90`) |

---

## 4. Kısa Değerlendirme

- **ID Test Seti (300 Görsel, `443` GT):** KNN/OOD filtresi yanlış pozitifleri `58` → `43` (`-%25.86`) düşürerek Precision'ı `+2.86` puan (`%86.88` → `%89.74`), `F1-Score`'u `IoU >= 0.50` seviyesinde `+0.46` puan (`%86.78` → `%87.24`) ve `IoU >= 0.75` seviyesinde `+0.99` puan (`%80.68` → `%81.67`) artırmıştır. Buna karşılık `8` gerçek domuz kutusunun elenmesi nedeniyle Recall `-1.81` puan (`%86.68` → `%84.88`), standart `mAP@0.5 (conf = 0.001)` ise `0.921035` → `0.895723` (`-0.025312`) olmuştur.
- **Karma Test Seti (378 Görsel, `130` GT + `298` OOD/Boş Görsel):** Yabancı türler (`wild_goat`, `equus_quagga`, `loxodonta_africana`) ve boş kareler eklendiğinde Ham Model A'nın ürettiği `205` yanlış pozitif, KNN/OOD filtresiyle `33`'e (`-%83.90`) inmiştir; özellikle `wild_goat` (`111` → `14`, `-%87.39`) ve `loxodonta_africana` (`33` → `2`, `-%93.94`) üzerinde güçlü OOD bastırması sağlanmıştır. Yalnızca `2` TP kaybedilirken (`108` → `106`, Recall `-1.54` puan), Precision `+41.75` puan (`%34.50` → `%76.26`), `F1-Score (IoU >= 0.50)` `+30.05` puan (`%48.76` → `%78.81`) ve standart `mAP@0.5 (conf = 0.001)` `+0.188752` (`0.641030` → `0.829782`) yükselmiştir.
