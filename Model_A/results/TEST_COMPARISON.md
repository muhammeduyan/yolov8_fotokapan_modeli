# Model A Test Karşılaştırma Raporu (Ham YOLOv8n vs. Temiz Çift Kapılı OOD Filtreli)

Bu rapor, **Ham Model A (KNN/OOD olmadan YOLOv8n)** ile **Temiz Bağımsız Çift Kapılı OOD Filtreli Model A (Letterbox +%10 Bağlam + Saf ResNet-50: Kapı 1 Tek-Sınıf Domuz OOD ∧ Kapı 2 Arka Plan L2 Lojistik Regresyon)** modellerinin hem **300 görsellik ID test seti** (`v0.3_wild_boar_closeup/images/test`) hem de **378 görsellik karma test seti** (`test/`) üzerindeki karşılaştırmalı sonuçlarını içerir.

---

## 1. Test Kurulumu

| Parametre | Değer |
|---|---|
| **ID Test Veri Seti** | `v0.3_wild_boar_closeup/images/test` (`300` görsel, **`443`** `wild_boar` kutusu) |
| **Karma Test Veri Seti** | `test/` (`378` görsel: `80` `wild_boar` [`130` GT] + `80` `wild_goat` + `79` `equus_quagga` + `79` `loxodonta_africana` + `60` `empty`) |
| **Dedektör Modeli** | `Model_A/model/best.pt` (YOLOv8n) |
| **OOD Referans Bankası** | `Model_A/model/knn_bank.npz` (`1.811` domuz kutusu, `1.800` `IoU = 0` arka plan yaması, sıfır kara liste / sıfır ImageNet sınıf kuralı) |
| **Kırpma & Öznitelik Uzayı** | Letterbox (`+%10` bağlam payı, `224×224`, en-boy oranı korumalı) + saf `2048`-D ResNet-50 |
| **Karar Mekanizması (Çift Kapı)** | **Kapı 1 (Saf Domuz OOD):** `[knn_boar, -pca_residual, -maha_dist]` (`U_tr` LOO standartlaştırmalı, `knn_bg` görmez) **∧** **Kapı 2 (Arka Plan):** L2 Lojistik Regresyon (`w = [1.4653, 4.7882, 2.0583, 3.4962]`, `b = -1.9230`); ortak skor `min(m_ood, m_bg)` |
| **Otomatik Öğrenilen Parametreler** | `k = 5`, `PCA p = 168` (`%90` varyans), OAS `alpha = 0.009071` |
| **Kalibrasyon Eşiği (`tau`)** | `0.5000` (yalnızca `val` setinden otomatik ortak TPR-95 eşiği, `percentile = 5.0`) |
| **Çalışma Noktası / Standart mAP** | Kutu metrikleri `conf >= 0.25`; standart `mAP@0.5` ve `mAP@0.5:0.95` ise `conf = 0.001` |
| **Donanım / Cihaz** | Apple Silicon **`mps`** |

---

## 2. Ana Karşılaştırma Tablosu (Test Seti — 300 Görsel, 443 GT Kutusu)

| Metrik | Ham Model A (KNN/OOD Yok) | Temiz Çift Kapılı OOD Filtreli (`tau=0.5000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `442` | `412` | **`-30`** (`-%6.79`) |
| **True Positives (TP, `IoU >= 0.50`)** | `384` | `379` | `-5` |
| **False Positives (FP, `IoU >= 0.50`)** | `58` | `33` | **`-25`** (`-%43.10`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `59` | `64` | `+5` |
| **Precision (`IoU >= 0.50`)** | `0.868778` (`%86.88`) | `0.919903` (`%91.99`) | **`+0.051125`** (`+5.11` puan) |
| **Recall (`IoU >= 0.50`)** | `0.866817` (`%86.68`) | `0.855530` (`%85.55`) | `-0.011287` (`-1.13` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.867797` (`%86.78`) | `0.886550` (`%88.66`) | **`+0.018753`** (`+1.88` puan) |
| **Precision (`IoU >= 0.75`)** | `0.807692` (`%80.77`) | `0.859223` (`%85.92`) | **`+0.051531`** (`+5.15` puan) |
| **Recall (`IoU >= 0.75`)** | `0.805869` (`%80.59`) | `0.799097` (`%79.91`) | `-0.006772` (`-0.68` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.806780` (`%80.68`) | `0.828070` (`%82.81`) | **`+0.021290`** (`+2.13` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.860887` | `0.852049` | `-0.008838` |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.921035` | `0.918737` | `-0.002298` |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.749471` | `0.746996` | `-0.002475` |

> *Not: Tek sınıflı (`wild_boar`) modelde `mAP@0.5` ile `AP@50` aynı metriktir. Ham Model A için Ultralytics `yolo val (split=test)` çapraz doğrulaması `mAP@0.5 = 0.923150`, `mAP@0.5:0.95 = 0.745972` olarak doğrulanmıştır.*

---

## 3. Ana Karşılaştırma Tablosu (Karma Test Seti — 378 Görsel, 130 GT `wild_boar` Kutusu)

> Test seti: `80` `wild_boar` + `80` `wild_goat` + `79` `equus_quagga` + `79` `loxodonta_africana` + `60` `empty` (toplam `130` adet GT `wild_boar` kutusu; diğer 298 görselde `wild_boar` GT'si `0`'dır).

| Metrik | Ham Model A (KNN/OOD Yok) | Temiz Çift Kapılı OOD Filtreli (`tau=0.5000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `313` | `138` | **`-175`** (`-%55.91`) |
| **True Positives (TP, `IoU >= 0.50`)** | `108` | `106` | `-2` |
| **False Positives (FP, `IoU >= 0.50`)** | `205` | `32` | **`-173`** (`-%84.39`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `22` | `24` | `+2` |
| **Precision (`IoU >= 0.50`)** | `0.345048` (`%34.50`) | `0.768116` (`%76.81`) | **`+0.423068`** (`+42.31` puan) |
| **Recall (`IoU >= 0.50`)** | `0.830769` (`%83.08`) | `0.815385` (`%81.54`) | `-0.015384` (`-1.54` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.487585` (`%48.76`) | `0.791045` (`%79.10`) | **`+0.303460`** (`+30.35` puan) |
| **Precision (`IoU >= 0.75`)** | `0.309904` (`%30.99`) | `0.695652` (`%69.57`) | **`+0.385748`** (`+38.57` puan) |
| **Recall (`IoU >= 0.75`)** | `0.746154` (`%74.62`) | `0.738462` (`%73.85`) | `-0.007692` (`-0.77` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.437923` (`%43.79`) | `0.716418` (`%71.64`) | **`+0.278495`** (`+27.85` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.622852` | `0.775025` | **`+0.152173`** |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.641030` | `0.815318` | **`+0.174288`** |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.534643` | `0.663081` | **`+0.128438`** |

### Kategori Bazlı Dağılım (378 Görsel, `conf >= 0.25`)

| Kategori | Görsel Sayısı | Ham Model A (Tespitli Görsel / FP) | Temiz Çift Kapılı OOD (Tespitli Görsel / FP) | FP Azalması |
|---|---:|---:|---:|---:|
| `wild_boar` | `80` | `80` (`%100.00`) / `18` (`108` TP) | `80` (`%100.00`) / `8` (`106` TP) | **`-10`** (`-%55.56`) |
| `wild_goat` | `80` | `66` (`%82.50`) / `111` | `11` (`%13.75`) / `15` | **`-96`** (`-%86.49`) |
| `equus_quagga` | `79` | `21` (`%26.58`) / `31` | `4` (`%5.06`) / `4` | **`-27`** (`-%87.10`) |
| `loxodonta_africana` | `79` | `28` (`%35.44`) / `33` | `2` (`%2.53`) / `2` | **`-31`** (`-%93.94`) |
| `empty` | `60` | `8` (`%13.33`) / `12` | `3` (`%5.00`) / `3` | **`-9`** (`-%75.00`) |
| **Toplam** | **`378`** | `203` / `205` (`108` TP) | `100` / `32` (`106` TP) | **`-173`** (`-%84.39`) |

---

## 4. Kısa Değerlendirme

- **ID Test Seti (300 Görsel, `443` GT):** Tamamen sınıf-etiketsiz, kara listesiz ve Letterbox (+%10 bağlam) tabanlı bağımsız çift kapılı OOD filtresi, yanlış pozitifleri `58` → `33` (`-%43.10`) düşürerek Precision'ı `+5.11` puan (`%86.88` → `%91.99`), `F1-Score`'u `IoU >= 0.50` seviyesinde `+1.88` puan (`%86.78` → `%88.66`) ve `IoU >= 0.75` seviyesinde `+2.13` puan (`%80.68` → `%82.81`) yükseltmiştir. Görsel bazlı domuz isabet oranı ham modelle birebir aynı (`298/300`, `%99.33`) kalırken standart `mAP@0.5` (`0.9210` → `0.9187`) ve `mAP@0.5:0.95` (`0.7495` → `0.7470`) neredeyse sıfır kayıpla korunmuştur.
- **Karma Test Seti (378 Görsel, `130` GT + `298` OOD/Boş):** Hiçbir yabancı hayvan kara listesi veya ImageNet sınıf ağırlığı kullanılmadan, **Kapı 1 (Saf Tek-Sınıf Domuz OOD)** ile **Kapı 2 (Arka Plan L2 Lojistik)** bağımsız `AND` (`min`) mantığıyla birleştirildiğinde yanlış pozitifler `205` → `32` (`-%84.39`) düşmüştür. `wild_boar` görsellerinde `%100` (`80/80`) tespit oranı ve `106/130` TP korunurken Precision `+42.31` puan (`%34.50` → `%76.81`), `F1-Score` `+30.35` puan (`%48.76` → `%79.10`) ve standart `mAP@0.5` `+0.1743` (`0.6410` → `0.8153`) artmıştır.

---

## 5. Metrikler Neyi Ölçer?

| Metrik / Terim | Neyi Ölçer? |
|---|---|
| **GT (Ground-Truth) Kutusu** | Görsellerde insan tarafından doğrulanmış gerçek yaban domuzu (`wild_boar`) etiket kutularıdır. |
| **IoU (*Intersection over Union*)** | Modelin çizdiği kutu ile gerçek (GT) kutunun örtüşme oranıdır (`0` = hiç örtüşmüyor, `1` = birebir örtüşüyor). `IoU >= 0.50` standart eşleşme eşiği, `IoU >= 0.75` ise daha sıkı konum doğruluğu testidir. |
| **Güven Eşiği (`conf`)** | YOLO'nun bir kutuyu tahmin olarak kabul etmesi için gereken minimum güven skorudur (`conf >= 0.25` gerçek kullanım çalışma noktası; `conf = 0.001` tüm aday kutuları kapsayan standart mAP hesabı). |
| **Kalibrasyon Eşiği (`tau = 0.5000`)** | Bağımsız Çift Kapı OOD filtresinin (Kapı 1: Saf Domuz OOD ∧ Kapı 2: Arka Plan L2 Lojistik, ortak skor `min(m_ood, m_bg)`) bir aday kutuyu "gerçek yaban domuzu" olarak onaylaması için gereken sigmoid skor eşiğidir (`val` setinin `%5` diliminden otomatik kalibre edilir). |
| **Toplam Tahmin Kutusu** | Modelin test setindeki tüm görsellerde `conf >= 0.25` üzerinde ürettiği toplam `wild_boar` kutu sayısıdır (`TP + FP`). |
| **True Positives (TP)** | Gerçek bir yaban domuzu kutusuyla belirtilen `IoU` eşiği üzerinde doğru eşleşen tahmin sayısıdır (*ne kadar yüksekse o kadar iyi*). |
| **False Positives (FP)** | Yanlış alarm sayısıdır: arka planı, başka bir hayvanı (keçi/zebra/fil) veya yanlış konumu `wild_boar` olarak işaretleme (*ne kadar düşükse o kadar iyi*). |
| **False Negatives (FN)** | Modelin tespit edemediği veya filtrenin yanlışlıkla elediği gerçek yaban domuzu sayısıdır (*ne kadar düşükse o kadar iyi*). |
| **Precision (Kesinlik)** | `TP / (TP + FP)` — Model "yaban domuzu var" dediğinde bunun yüzde kaçının gerçekten yaban domuzu olduğunu ölçer. Yüksek Precision, az yanlış alarm demektir. |
| **Recall (Duyarlılık)** | `TP / (TP + FN)` — Görsellerdeki tüm gerçek yaban domuzlarının yüzde kaçının yakalanabildiğini ölçer. Yüksek Recall, az domuz kaçırmak demektir. |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` — Precision ve Recall'un harmonik ortalamasıdır; yanlış alarm azaltma ile gerçek domuz yakalama arasındaki genel dengeyi tek skorda özetler. |
| **mAP@0.5 (`conf >= 0.25`)** | `IoU >= 0.50` eşiğinde, yalnızca gerçek çalışma noktasındaki (`conf >= 0.25`) tahminlerle oluşturulan Precision–Recall eğrisi altındaki alandır (tek sınıfta `AP@50` ile aynıdır). |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | Ultralytics/COCO standardına uygun olarak neredeyse tüm aday kutular (`conf = 0.001`) dahil edilerek `IoU = 0.50` eşiğinde hesaplanan ortalama kesinliktir. |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `IoU = 0.50`'den `0.95`'e kadar `0.05` aralıkla 10 farklı örtüşme eşiğinde hesaplanan `mAP` değerlerinin ortalamasıdır; hem doğru tespiti hem de kutunun nesneye tam oturma hassasiyetini ölçer. |
| **FP Azalması** | OOD filtresi uygulandıktan sonra ham modele kıyasla kaç yanlış pozitifin elendiğini ve yüzdesel düşüş oranını gösterir. |
