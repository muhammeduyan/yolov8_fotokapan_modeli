# Fotokapan Görüntülerinde Yaban Hayatı Tespiti

Bu depo, fotokapan (camera trap) görüntülerinde yaban hayatı tespiti ve dağılım dışı (OOD) yanlış alarm filtreleme için geliştirilen modelleri içerir.

## Modeller

| Klasör | Açıklama | Durum |
|---|---|---|
| **[Model_A/](Model_A/README.md)** | Yerel Yaban Domuzu (`wild_boar` v0.4) — YOLOv8n + ResNet-50 Çift Bankalı KNN & PCA OOD Filtresi | Aktif |
| **[Model_B/](Model_B/README.md)** | Dört sınıflı YOLOv8n + BioCLIP/KNN/learned gate/SpeciesNet OOD zinciri | Hazırlandı |

## Hızlı Başlangıç (`Model_A`)

```bash
git clone https://github.com/muhammeduyan/yolov8_fotokapan_modeli.git
cd yolov8_fotokapan_modeli/Model_A

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./run_demo.sh
```

Ayrıntılı mimari, benchmark sonuçları ve eğitim komutları için **[Model_A/README.md](Model_A/README.md)** dosyasına bakabilirsiniz.
