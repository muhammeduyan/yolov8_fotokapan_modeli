# Model A Test Karşılaştırma Raporu (Ham YOLOv8n vs. Temiz KNN/OOD Filtreli)

Bu rapor, **Ham Model A (KNN/OOD olmadan YOLOv8n)** ile **Temiz KNN/OOD Filtreli Model A (Letterbox +%10 Bağlam + Saf ResNet-50 Çift Bankalı k-NN / PCA / OAS Mahalanobis + L2 Lojistik Regresyon)** modellerinin hem **300 görsellik ID test seti** (`v0.3_wild_boar_closeup/images/test`) hem de **378 görsellik karma test seti** (`test/`) üzerindeki karşılaştırmalı sonuçlarını içerir.

---

## 1. Test Kurulumu

| Parametre | Değer |
|---|---|
| **ID Test Veri Seti** | `v0.3_wild_boar_closeup/images/test` (`300` görsel, **`443`** `wild_boar` kutusu) |
| **Karma Test Veri Seti** | `test/` (`378` görsel: `80` `wild_boar` [`130` GT] + `80` `wild_goat` + `79` `equus_quagga` + `79` `loxodonta_africana` + `60` `empty`) |
| **Dedektör Modeli** | `Model_A/model/best.pt` (YOLOv8n) |
| **OOD Referans Bankası** | `Model_A/model/knn_bank.npz` (`1.811` domuz kutusu, `1.800` `IoU = 0` arka plan yaması, sıfır kara liste / sıfır ImageNet sınıf kuralı) |
| **Kırpma & Öznitelik Uzayı** | Letterbox (`+%10` bağlam payı, `224×224`, en-boy oranı korumalı) + saf `2048`-D ResNet-50 |
| **Otomatik Öğrenilen Parametreler** | `k = 5`, `PCA p = 168` (`%90` varyans), OAS `alpha = 0.009071`, L2 Lojistik Regresyon (`w = [1.4653, 4.7882, 2.0583, 3.4962]`, `b = -1.9230`) |
| **Kalibrasyon Eşiği (`tau`)** | `0.5000` (yalnızca `val` setinden otomatik TPR-95 eşiği, `percentile = 5.0`) |
| **Çalışma Noktası / Standart mAP** | Kutu metrikleri `conf >= 0.25`; standart `mAP@0.5` ve `mAP@0.5:0.95` ise `conf = 0.001` |
| **Donanım / Cihaz** | Apple Silicon **`mps`** |

---

## 2. Ana Karşılaştırma Tablosu (Test Seti — 300 Görsel, 443 GT Kutusu)

| Metrik | Ham Model A (KNN/OOD Yok) | Temiz KNN/OOD Filtreli Model A (`tau=0.5000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `442` | `407` | **`-35`** (`-%7.92`) |
| **True Positives (TP, `IoU >= 0.50`)** | `384` | `377` | `-7` |
| **False Positives (FP, `IoU >= 0.50`)** | `58` | `30` | **`-28`** (`-%48.28`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `59` | `66` | `+7` |
| **Precision (`IoU >= 0.50`)** | `0.868778` (`%86.88`) | `0.926290` (`%92.63`) | **`+0.057512`** (`+5.75` puan) |
| **Recall (`IoU >= 0.50`)** | `0.866817` (`%86.68`) | `0.851016` (`%85.10`) | `-0.015801` (`-1.58` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.867797` (`%86.78`) | `0.887059` (`%88.71`) | **`+0.019262`** (`+1.93` puan) |
| **Precision (`IoU >= 0.75`)** | `0.807692` (`%80.77`) | `0.864865` (`%86.49`) | **`+0.057173`** (`+5.72` puan) |
| **Recall (`IoU >= 0.75`)** | `0.805869` (`%80.59`) | `0.794582` (`%79.46`) | `-0.011287` (`-1.13` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.806780` (`%80.68`) | `0.828235` (`%82.82`) | **`+0.021455`** (`+2.15` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.860887` | `0.847793` | `-0.013094` |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.921035` | `0.915817` | `-0.005218` |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.749471` | `0.747052` | `-0.002419` |

> *Not: Tek sınıflı (`wild_boar`) modelde `mAP@0.5` ile `AP@50` aynı metriktir. Ham Model A için Ultralytics `yolo val (split=test)` çapraz doğrulaması `mAP@0.5 = 0.923150`, `mAP@0.5:0.95 = 0.745972` olarak doğrulanmıştır.*

---

## 3. Ana Karşılaştırma Tablosu (Karma Test Seti — 378 Görsel, 130 GT `wild_boar` Kutusu)

> Test seti: `80` `wild_boar` + `80` `wild_goat` + `79` `equus_quagga` + `79` `loxodonta_africana` + `60` `empty` (toplam `130` adet GT `wild_boar` kutusu; diğer 298 görselde `wild_boar` GT'si `0`'dır).

| Metrik | Ham Model A (KNN/OOD Yok) | Temiz KNN/OOD Filtreli Model A (`tau=0.5000`) | Fark (Δ) |
|---|---:|---:|---:|
| **Toplam Tahmin Kutusu (`conf >= 0.25`)** | `313` | `165` | **`-148`** (`-%47.28`) |
| **True Positives (TP, `IoU >= 0.50`)** | `108` | `105` | `-3` |
| **False Positives (FP, `IoU >= 0.50`)** | `205` | `60` | **`-145`** (`-%70.73`) |
| **False Negatives (FN, `IoU >= 0.50`)** | `22` | `25` | `+3` |
| **Precision (`IoU >= 0.50`)** | `0.345048` (`%34.50`) | `0.636364` (`%63.64`) | **`+0.291316`** (`+29.13` puan) |
| **Recall (`IoU >= 0.50`)** | `0.830769` (`%83.08`) | `0.807692` (`%80.77`) | `-0.023077` (`-2.31` puan) |
| **F1-Score (`IoU >= 0.50`)** | `0.487585` (`%48.76`) | `0.711864` (`%71.19`) | **`+0.224279`** (`+22.43` puan) |
| **Precision (`IoU >= 0.75`)** | `0.309904` (`%30.99`) | `0.575758` (`%57.58`) | **`+0.265854`** (`+26.59` puan) |
| **Recall (`IoU >= 0.75`)** | `0.746154` (`%74.62`) | `0.730769` (`%73.08`) | `-0.015385` (`-1.54` puan) |
| **F1-Score (`IoU >= 0.75`)** | `0.437923` (`%43.79`) | `0.644068` (`%64.41`) | **`+0.206145`** (`+20.61` puan) |
| **mAP@0.5 (`conf >= 0.25` çalışma noktası)** | `0.622852` | `0.737038` | **`+0.114186`** |
| **mAP@0.5 (`conf = 0.001` standart YOLO)** | `0.641030` | `0.770509` | **`+0.129479`** |
| **mAP@0.5:0.95 (`conf = 0.001` standart YOLO)** | `0.534643` | `0.631635` | **`+0.096992`** |

### Kategori Bazlı Dağılım (378 Görsel, `conf >= 0.25`)

| Kategori | Görsel Sayısı | Ham Model A (Tespitli Görsel / FP) | Temiz KNN/OOD Filtreli (Tespitli Görsel / FP) | FP Azalması |
|---|---:|---:|---:|---:|
| `wild_boar` | `80` | `80` (`%100.00`) / `18` (`108` TP) | `80` (`%100.00`) / `8` (`105` TP) | **`-10`** (`-%55.56`) |
| `wild_goat` | `80` | `66` (`%82.50`) / `111` | `25` (`%31.25`) / `33` | **`-78`** (`-%70.27`) |
| `equus_quagga` | `79` | `21` (`%26.58`) / `31` | `4` (`%5.06`) / `4` | **`-27`** (`-%87.10`) |
| `loxodonta_africana` | `79` | `28` (`%35.44`) / `33` | `11` (`%13.92`) / `12` | **`-21`** (`-%63.64`) |
| `empty` | `60` | `8` (`%13.33`) / `12` | `3` (`%5.00`) / `3` | **`-9`** (`-%75.00`) |
| **Toplam** | **`378`** | `203` / `205` (`108` TP) | `123` / `60` (`105` TP) | **`-145`** (`-%70.73`) |

---

## 4. Kısa Değerlendirme

- **ID Test Seti (300 Görsel, `443` GT):** Tamamen sınıf-etiketsiz, kara listesiz ve Letterbox (+%10 bağlam) tabanlı yeni OOD filtresi, yanlış pozitifleri `58` → `30` (`-%48.28`) düşürerek Precision'ı `+5.75` puan (`%86.88` → `%92.63`), `F1-Score`'u `IoU >= 0.50` seviyesinde `+1.93` puan (`%86.78` → `%88.71`) ve `IoU >= 0.75` seviyesinde `+2.15` puan (`%80.68` → `%82.82`) yükseltmiştir. Standart `mAP@0.5` (`0.9210` → `0.9158`) ve `mAP@0.5:0.95` (`0.7495` → `0.7471`) neredeyse hiç kayıp vermeden korunmuştur.
- **Karma Test Seti (378 Görsel, `130` GT + `298` OOD/Boş):** Hiçbir yabancı hayvan kara listesi veya ImageNet sınıf ağırlığı kullanılmadan, yalnızca eğitim domuz kutuları ve kendi `IoU = 0` arka plan yamalarından öğrenilen filtre yanlış pozitifleri `205` → `60` (`-%70.73`) indirmiş; `wild_boar` görsellerinde `%100` (`80/80`) tespit oranını korurken Precision'ı `+29.13` puan (`%34.50` → `%63.64`), `F1-Score`'u `+22.43` puan (`%48.76` → `%71.19`) ve standart `mAP@0.5`'i `+0.1295` (`0.6410` → `0.7705`) yükseltmiştir.

---

## 5. Metrikler Neyi Ölçer?

| Metrik / Terim | Neyi Ölçer? |
|---|---|
| **GT (Ground-Truth) Kutusu** | Görsellerde insan tarafından doğrulanmış gerçek yaban domuzu (`wild_boar`) etiket kutularıdır. |
| **IoU (*Intersection over Union*)** | Modelin çizdiği kutu ile gerçek (GT) kutunun örtüşme oranıdır (`0` = hiç örtüşmüyor, `1` = birebir örtüşüyor). `IoU >= 0.50` standart eşleşme eşiği, `IoU >= 0.75` ise daha sıkı konum doğruluğu testidir. |
| **Güven Eşiği (`conf`)** | YOLO'nun bir kutuyu tahmin olarak kabul etmesi için gereken minimum güven skorudur (`conf >= 0.25` gerçek kullanım çalışma noktası; `conf = 0.001` tüm aday kutuları kapsayan standart mAP hesabı). |
| **Kalibrasyon Eşiği (`tau = 0.5000`)** | Saf ResNet-50 OOD filtresinin (k-NN + `%90` varyans PCA + OAS Mahalanobis + L2 Lojistik Regresyon) bir aday kutuyu "gerçek yaban domuzu" olarak onaylaması için gereken sigmoid skor eşiğidir (`val` setinin `%5` diliminden otomatik kalibre edilir). |
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
