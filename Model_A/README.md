# Model A — Fotokapan Görüntülerinde Yaban Domuzu Tespiti ve OOD Filtreleme (`v0.4`)

Bu proje, yerel fotokapan (camera trap) görüntülerinde **yaban domuzu (`wild_boar`)** tespiti yapan ve tek sınıflı dedektörlerin yaşadığı **dağılım dışı (OOD) hayvan / kaya-ağaç arka plan yanlış alarmlarını** bastıran iki aşamalı bir bilgisayarlı görü sistemidir.

- **1. Aşama (Tespit):** Yakın çekim (close-up) ve boş arka plan görselleriyle ince ayar yapılmış **YOLOv8n** (`model/best.pt`)
- **2. Aşama (OOD & Arka Plan Filtresi):** Dondurulmuş **ResNet-50** öznitelik uzayında **Çift Bankalı KNN + 140-D PCA Alt-Uzayı + TPR-95 Otomatik Kalibrasyon** (`model/knn_bank.npz`)

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
> **Not:** Eğitilmiş model ağırlığı (`model/best.pt`, 6,2 MB) ve kalibre edilmiş referans bankası (`model/knn_bank.npz`, 13 MB) repoda hazırdır. Tahmin ve demo için 4,6 GB'lık eğitim veri setini indirmeniz gerekmez.

### Demo, Tahmin ve Benchmark
```bash
# 1) Hazır demo (keçi OOD testi + gerçek domuz + 100 negatif benchmark)
./run_demo.sh

# 2) Tekil görsel tahmini (ham ve KNN-filtreli çıktılar outputs/knn_predictions/ altına kaydedilir)
python src/predict.py --source samples/01_wild_boar.jpg
python src/predict.py --source samples/02_same_goat.jpg

# 3) Kalibrasyon ve 100 negatif OOD benchmark raporu
python src/predict.py --calibrate
python src/predict.py --benchmark
```

### Yeniden Eğitim (Harici Veri Seti ile)
4,6 GB boyutundaki `v0.3_wild_boar_closeup` veri seti GitHub boyut limitleri nedeniyle depoya dahil edilmemiştir. Veri setini bir üst klasöre (`../v0.3_wild_boar_closeup/`) yerleştirip çalıştırabilirsiniz:

```bash
python src/train.py --data ../v0.3_wild_boar_closeup/data.yaml --epochs 8 --freeze 10 --lr0 0.0004
```

---

## 2. Modelin Gelişim Süreci (`v0.2 → v0.3 → v0.4`)

| Aşama | Veri Seti | Eğitim Yaklaşımı | Çözdüğü Problem |
|---|---|---|---|
| **1. Temel Model (`v0.2`)** | 1.200 train / 300 val / 300 test insan doğrulamalı yerel fotokapan görseli | `yolov8n.pt` tabanından 30 epoch | Sızıntısız (kaynak klasör + 5 sn sekans + kırpma bazlı) ilk yerel domuz dedektörü (`Test mAP@50: 0.924`). |
| **2. Yakın Çekim (`v0.3_closeup`)** | 1.200 orijinal + **600 sentetik close-up** = 1.800 train (`scripts/generate_closeup_crops.py`) | `yolov8n.pt` tabanından 60 epoch, `batch=16`, `imgsz=640` | Fotokapana çok yaklaşan veya kadrajda yarısı kalan domuzların kaçırılması önlendi (`Val mAP@50: 0.935`, `Recall: 0.893`). |
| **3. Arka Plan İnce Ayarı (`v0.4`) ★ Final** | 1.800 görsel + **250 boş kaya/ağaç/çalı arka planı** = 2.050 train (`scripts/generate_background_and_bank.py`) | `v0.3` ağırlıkları üzerine **8 epoch fine-tune**, `freeze=10`, `lr0=0.0004` | Gece görüntülerindeki yüksek dokulu taş, kaya, ağaç ve çalı yanlış alarmları azaltıldı. |
| **4. Çift Bankalı OOD Filtresi** | `model/knn_bank.npz` (1.811 domuz + 1.738 arka plan yaması) | Dondurulmuş ResNet-50 öznitelik uzayında post-hoc kalibreli filtre | Yaban keçisi, zebra, fil gibi dağılım dışı (OOD) hayvanların `wild_boar` sanılmasını engeller. |

---

## 3. İç Mimari (İki Aşamalı Hibrit Pipeline)

```text
Fotokapan Görseli
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  1. AŞAMA: YOLOv8n Nesne Dedektörü (model/best.pt)       │
│  • Tek sınıf: wild_boar (imgsz=640, conf=0.25)           │
└──────────────────────────┬───────────────────────────────┘
                           │ (Her aday kutu kırpılır → 224×224)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  2. AŞAMA: ResNet-50 Çift Bankalı KNN & Alt-Uzay Filtresi │
│  (src/detector.py + model/knn_bank.npz)                  │
│  1. Domuz Bankası KNN (U_tr: 1.811×2048, k=5)            │
│  2. Arka Plan Bankası KNN (U_bg: 1.738×2048, k=5)        │
│  3. Ön Plan / Arka Plan Ayırıcı İzdüşümü (w_fg)          │
│  4. 140-Boyutlu PCA Domuz Alt-Uzayı Artık Hatası (Vp)    │
│  5. Morfolojik Domuz (Sus scrofa) Uyumu & OOD Cezası     │
│  6. Çift Kapı (g1/g2) + TPR-95 Eşiği (tau = 0.7000)      │
│  7. Sürü Bütünlüğü (Herd Context) & Kenar Desteği        │
└──────────────────────────┬───────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    ✅ wild_boar (Onaylandı)      🚫 OOD_suppressed (Elendi)
```

---

## 4. Performans ve Doğrulanmış Sonuçlar

| Bileşen / Ölçüm | Değer |
|---|---:|
| **YOLOv8n Val Precision / Recall** (300 görsel, 457 kutu) | `0.964` / `0.873` |
| **YOLOv8n Val mAP@50 / mAP@50-95** | `0.929` / `0.728` |
| **KNN Otomatik Kalibrasyon Eşiği (`tau_95`)** | `0.7000` |
| **Val / Test Gerçek Domuz Koruma Oranı (TPR)** | `%94.31` (`431/457`) / **`%94.36` (`418/443`)** |
| **Kaya/Ağaç Arka Plan Reddetme Oranı** | **`%97.70`** (`1.738` yama) |
| **100 Negatif OOD Challenge FP Azaltımı** | **`43/100 → 6/100` (`%86.0` azalma)** |
| — Boş Fotokapan Kareleri / Afrika Fili (FP) | `7/40 → 0/40` / `13/20 → 0/20` |
| — Zebra / Yaban Keçisi (FP) | `5/20 → 2/20` / `18/20 → 4/20` |

Eğitim eğrileri ve karmaşıklık matrisi `results/` klasöründedir (`results.png`, `confusion_matrix_normalized.png`, `results.csv`, `args.yaml`).

---

## 5. Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| **PyTorch (`torch`)** | Model eğitimi, tensör işlemleri ve Apple Silicon **MPS** / **CUDA** hızlandırma |
| **Ultralytics YOLOv8 (`yolov8n`)** | Tek sınıflı `wild_boar` nesne tespiti ve arka plan ince ayarı |
| **Torchvision ResNet-50** | Aday kutulardan 2048-D öznitelik ve ImageNet *Sus scrofa* / OOD logit marjı çıkarımı |
| **OpenCV & Pillow** | Görsel okuma, kırpma, Laplacian netlik filtresi ve tespit kutusu çizimi |
| **NumPy & Pandas & PyYAML** | k-NN (`k=5`), 140-D PCA alt-uzayı, TPR-95 kalibrasyon, manifest ve YAML yönetimi |

---

## 6. Proje Yapısı

```text
Model_A/
├── src/
│   ├── __init__.py
│   ├── detector.py                    # YOLO + ResNet-50 Çift Bankalı KNN/OOD motoru
│   ├── predict.py                     # Tahmin, kalibrasyon ve benchmark CLI aracı
│   └── train.py                       # YOLOv8n eğitim / fine-tuning betiği
├── model/
│   ├── best.pt                        # Final YOLOv8n ağırlığı (6,2 MB)
│   ├── knn_bank.npz                   # Kalibre edilmiş domuz & arka plan bankası (13 MB)
│   └── benchmark_manifest.csv         # 100 negatif OOD benchmark manifesti
├── results/
│   ├── results.png                    # Eğitim kayıp & metrik grafikleri
│   ├── confusion_matrix_normalized.png# Normalize karmaşıklık matrisi
│   ├── results.csv                    # Epoch bazlı sayısal metrikler
│   └── args.yaml                      # Eğitim hiperparametreleri
├── samples/                           # Hızlı test görselleri (domuz, keçi, zebra, fil, boş)
├── scripts/
│   ├── generate_closeup_crops.py      # Sentetik yakın çekim veri artırma aracı
│   └── generate_background_and_bank.py# Boş arka plan ve KNN yama üretici
├── outputs/                           # Üretilen tahmin görselleri (.gitignore ile hariç)
├── run_demo.sh                        # Tek komutluk demo ve benchmark scripti
├── README.md                          # Proje dokümantasyonu
└── requirements.txt                   # Python bağımlılıkları
```
