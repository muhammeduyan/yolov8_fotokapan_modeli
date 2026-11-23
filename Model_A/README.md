# Model A — Fotokapan Görüntülerinde Yaban Domuzu Tespiti ve OOD Filtreleme (`v0.4`)

Bu proje, yerel fotokapan (camera trap) görüntülerinde **yaban domuzu (`wild_boar`)** tespiti yapan ve tek sınıflı dedektörlerin yaşadığı **dağılım dışı (OOD) hayvan / kaya-ağaç arka plan yanlış alarmlarını** eğitim verisine başka hayvan sınıfı eklemeden bastıran iki aşamalı bir bilgisayarlı görü sistemidir.

- **1. Aşama (Tespit):** Yakın çekim (close-up) ve boş arka plan görselleriyle ince ayar yapılmış **YOLOv8n** (`model/best.pt`)
- **2. Aşama (Temiz Bağımsız Çift Kapılı OOD Filtresi):** **Letterbox (`+%10` bağlam payı)** + saf `2048`-D **ResNet-50** öznitelik uzayında **Kapı 1 (Saf Tek-Sınıf Domuz OOD: Deep k-NN + `%90` PCA + OAS Mahalanobis) ∧ Kapı 2 (Arka Plan L2 Lojistik Regresyon)** + ortak `val` **TPR-95 Eşikleme** (`model/knn_bank.npz`) — *sıfır hayvan kara listesi, sıfır ImageNet sınıf kuralı, sıfır elle ayarlanmış katsayı.*

---

## 1. Hızlı Kullanım

### Kurulum
```bash
git clone https://github.com/muhammeduyan/yolov8_fotokapan_modeli.git
cd yolov8_fotokapan_modeli/Model_A

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> **Not:** Eğitilmiş model ağırlığı (`model/best.pt`, 6,2 MB) ve sıfır sızıntıyla kalibre edilmiş referans bankası (`model/knn_bank.npz`, 27 MB) repoda hazırdır. Tahmin ve demo için 4,6 GB'lık eğitim veri setini indirmeniz gerekmez.

### Demo, Tahmin ve Kalibrasyon Raporu
```bash
# 1) Hazır demo (keçi OOD testi + gerçek domuz + kalibrasyon raporu)
./run_demo.sh

# 2) Tekil görsel tahmini (ham ve OOD-filtreli çıktılar outputs/knn_predictions/ altına kaydedilir)
python src/predict.py --source samples/01_wild_boar.jpg
python src/predict.py --source samples/02_same_goat.jpg

# 3) Kalibrasyon ve otomatik öğrenilen parametre raporu
python src/predict.py --calibrate
```

### Referans Bankasını Sıfırdan Üretme ve Yeniden Eğitim (Harici Veri Seti ile)
4,6 GB boyutundaki `v0.3_wild_boar_closeup` veri seti GitHub boyut limitleri nedeniyle depoya dahil edilmemiştir. Veri setini bir üst klasöre (`../v0.3_wild_boar_closeup/`) yerleştirip hem referans bankasını hem modeli sıfırdan yeniden üretebilirsiniz:

```bash
# Referans bankasını (knn_bank.npz) deterministik olarak (seed=42) sıfırdan üret:
python src/build_bank.py --data ../v0.3_wild_boar_closeup

# YOLOv8n modelini ince ayar yap:
python src/train.py --data ../v0.3_wild_boar_closeup/data.yaml --epochs 8 --freeze 10 --lr0 0.0004
```

---

## 2. Modelin Gelişim Süreci (`v0.2 → v0.3 → v0.4 → Çift Kapılı Temiz OOD`)

| Aşama | Veri Seti | Eğitim / Kalibrasyon Yaklaşımı | Çözdüğü Problem |
|---|---|---|---|
| **1. Temel Model (`v0.2`)** | 1.200 train / 300 val / 300 test insan doğrulamalı yerel fotokapan görseli | `yolov8n.pt` tabanından 30 epoch | Sızıntısız (kaynak klasör + 5 sn sekans + kırpma bazlı) ilk yerel domuz dedektörü (`Test mAP@50: 0.924`). |
| **2. Yakın Çekim (`v0.3_closeup`)** | 1.200 orijinal + **600 sentetik close-up** = 1.800 train (`scripts/generate_closeup_crops.py`) | `yolov8n.pt` tabanından 60 epoch, `batch=16`, `imgsz=640` | Fotokapana çok yaklaşan veya kadrajda yarısı kalan domuzların kaçırılması önlendi (`Val mAP@50: 0.935`, `Recall: 0.893`). |
| **3. Arka Plan İnce Ayarı (`v0.4`) ★ Final** | 1.800 görsel + **250 boş kaya/ağaç/çalı arka planı** = 2.050 train (`scripts/generate_background_and_bank.py`) | `v0.3` ağırlıkları üzerine **8 epoch fine-tune**, `freeze=10`, `lr0=0.0004` | Gece görüntülerindeki yüksek dokulu taş, kaya, ağaç ve çalı yanlış alarmları azaltıldı. |
| **4. Bağımsız Çift Kapılı Temiz OOD Filtresi** | `model/knn_bank.npz` (`1.811` domuz + `1.800` `IoU=0` arka plan yaması, `src/build_bank.py`) | Letterbox (`+%10`) + saf ResNet-50: **Kapı 1** (Saf Domuz OOD) **∧** **Kapı 2** (Arka Plan L2 Lojistik) + `val` ortak TPR-95 | Hiçbir yabancı hayvan verisi veya kara liste olmadan hem arka plan hem dağılım dışı (OOD) hayvan yanlış alarmlarını bastırır. |

---

## 3. İç Mimari (İki Aşamalı Hibrit Pipeline)

```text
Fotokapan Görseli
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│  1. AŞAMA: YOLOv8n Nesne Dedektörü (model/best.pt)             │
│  • Tek sınıf: wild_boar (imgsz=640, conf=0.25)                 │
└──────────────────────────┬─────────────────────────────────────┘
                           │ (Letterbox +%10 bağlam → 224×224)
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  2. AŞAMA: Saf ResNet-50 Bağımsız Çift Kapılı OOD Filtresi     │
│  (src/detector.py + model/knn_bank.npz)                        │
│                                                                │
│  • KAPI 1 (Saf Tek-Sınıf Domuz OOD — knn_bg'yi hiç görmez):    │
│    1. Domuz Bankası Deep k-NN (U_tr: 1.811×2048, k=5)          │
│    2. Otomatik PCA Alt-Uzay Artık Hatası (p=168, %90 varyans)  │
│    3. OAS Büzülmeli Mahalanobis Uzaklığı (alpha=0.0091)        │
│                                                                │
│  • KAPI 2 (Arka Plan Reddi):                                   │
│    4. Domuz vs. IoU=0 Arka Plan (U_bg: 1.800×2048) L2 Lojistik │
│                                                                │
│  • ORTAK KARAR (AND): z = min(m_ood, m_bg), val TPR-95 (0.50)  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    ✅ wild_boar (Onaylandı)      🚫 OOD_suppressed (Elendi)
```

---

## 4. Performans ve Doğrulanmış Sonuçlar

| Bileşen / Ölçüm | Değer |
|---|---:|
| **OOD Otomatik Kalibrasyon Eşiği (`tau`, TPR-95)** | `0.5000` (`Val TPR: %94.97`, `Test yama TPR: %97.74`) |
| **Kaya/Ağaç Arka Plan Reddetme Oranı (`1.800` yama)** | **`%99.22`** (`1.786 / 1.800`) |
| **300 ID Test Seti (`443` GT) — Precision / Recall / F1** | **`%91.99`** / **`%85.55`** / **`%88.66`** *(Ham: `%86.88 / %86.68 / %86.78`)* |
| **300 ID Test Seti — FP / `mAP@0.5` (`conf=0.001`)** | `58 → 33` (`-%43.10`) / **`0.9187`** *(Ham: `0.9210`)* |
| **378 Karma Test Seti (`130` GT + `298` OOD/Boş) — FP** | **`205 → 32` (`-%84.39` azalma)** |
| **378 Karma Test Seti — Precision / F1 / `mAP@0.5`** | **`%76.81`** / **`%79.10`** / **`0.8153`** *(Ham: `%34.50 / %48.76 / 0.6410`)* |

Eğitim eğrileri ve karmaşıklık matrisi `results/` klasöründedir (`results.png`, `confusion_matrix_normalized.png`, `results.csv`, `args.yaml`).

> **Detaylı Test Karşılaştırma Raporu:** Hem 300 görsellik ID test seti hem 378 görsellik karma test seti üzerinde Ham Model A vs. Temiz Çift Kapılı OOD Filtreli Model A'nın tüm `TP/FP/FN`, `Precision`, `Recall`, `F1-Score`, `mAP@0.5`, `mAP@0.5:0.95` ve kategori bazlı sonuçları ile metrik sözlüğü için **[results/TEST_COMPARISON.md](results/TEST_COMPARISON.md)** dosyasına bakabilirsiniz.

---

## 5. Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| **PyTorch (`torch`)** | Model eğitimi, tensör işlemleri ve Apple Silicon **MPS** / **CUDA** hızlandırma |
| **Ultralytics YOLOv8 (`yolov8n`)** | Tek sınıflı `wild_boar` nesne tespiti ve arka plan ince ayarı |
| **Torchvision ResNet-50** | Aday kutulardan saf `2048`-D görsel öznitelik çıkarımı (sınıf logitleri kullanılmaz) |
| **OpenCV & Pillow** | Görsel okuma, Letterbox (+%10 bağlam) kırpma, Laplacian doku filtresi ve kutu çizimi |
| **NumPy & Pandas & PyYAML** | Deep k-NN (`k=5`), SVD PCA (`%90` varyans), OAS Mahalanobis, Newton-IRLS L2 Lojistik Regresyon ve bağımsız çift kapı TPR-95 kalibrasyonu |

---

## 6. Proje Yapısı

```text
Model_A/
├── src/
│   ├── __init__.py
│   ├── detector.py                    # Letterbox + YOLO + saf ResNet-50 çift kapılı OOD motoru
│   ├── build_bank.py                  # Sıfır sızıntılı knn_bank.npz oluşturucu
│   ├── predict.py                     # Tahmin ve kalibrasyon raporu CLI aracı
│   └── train.py                       # YOLOv8n eğitim / fine-tuning betiği
├── model/
│   ├── best.pt                        # Final YOLOv8n ağırlığı (6,2 MB)
│   ├── knn_bank.npz                   # Sıfırdan üretilmiş kalibre referans bankası (27 MB)
│   └── benchmark_manifest.csv         # Negatif OOD görsel manifesti
├── results/
│   ├── TEST_COMPARISON.md             # 300 ID + 378 karma test karşılaştırma raporu & metrik sözlüğü
│   ├── results.png                    # Eğitim kayıp & metrik grafikleri
│   ├── confusion_matrix_normalized.png# Normalize karmaşıklık matrisi
│   ├── results.csv                    # Epoch bazlı sayısal metrikler
│   └── args.yaml                      # Eğitim hiperparametreleri
├── samples/                           # Hızlı test görselleri (domuz, keçi, zebra, fil, boş)
├── scripts/
│   ├── generate_closeup_crops.py      # Sentetik yakın çekim veri artırma aracı
│   └── generate_background_and_bank.py# Boş arka plan üretici
├── outputs/                           # Üretilen tahmin görselleri (.gitignore ile hariç)
├── run_demo.sh                        # Tek komutluk demo scripti
├── README.md                          # Proje dokümantasyonu
└── requirements.txt                   # Python bağımlılıkları
```
