# Model B — Çok Sınıflı Fotokapan Tespiti ve OOD

Model B, fotokapan görüntülerinde dört sınıfı YOLOv8n ile tespit eder:

| ID | Sınıf |
|---:|---|
| 0 | `wild_boar` |
| 1 | `wild_goat` |
| 2 | `equus_quagga` |
| 3 | `loxodonta_africana` |

YOLO tespitinden sonra isteğe bağlı OOD zinciri; BioCLIP özellikleri, KNN referans bankası, learned gate, SpeciesNet ve E2 → E3 → E4 → E5-B karar kurallarını kullanır. OOD kodu `src/phototrap_ood/` ve `src/model_b_ood/runtime/` altındadır.

## Kurulum

PowerShell:

```powershell
cd Model_B
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-ood.txt
```

CUDA kullanan PyTorch kurulumu makineye göre ayrıca seçilmelidir. `requirements.txt` içindeki `torch` satırı CUDA sürümünü sabitlemez.

## Ağırlıklar ve veri

YOLO ağırlığı `models/multispecies_v0.3_demo/best.pt` altındadır. OOD için gereken BioCLIP, SpeciesNet ve referans bankası `models/ood/e5b_v1/` altındadır.

Bu dosyalar Git LFS ile izlenecek şekilde yapılandırılmıştır. OOD varlıkları yaklaşık 858 MB olduğundan Git LFS kurulmadan normal Git ile eklenmemelidir.

Ham görüntüler ve label dosyaları depoya eklenmez. Eğitim/değerlendirme için `data/prepared/v0.3_multispecies_demo_clean_v2/` altına veri setinin görüntü ve label klasörleri yerleştirilmelidir. `data.yaml`, manifest ve doğrulama kayıtları bu pakette bulunur.

## Veri kontrolü

```powershell
python prepare_data.py --help
python prepare_data.py --model b
```

## Eğitim

```powershell
python train.py --model b --device auto --name model_b_clean_v2
```

Komut `configs/model_b.yaml` kullanır ve mevcut run üzerine yazmaz.

## Değerlendirme

```powershell
python evaluate.py --model b --device auto
```

Config içindeki `evaluation.split` değeri `test` olarak ayarlanmıştır. Test ve validation sonuçları karıştırılmamalıdır.

## YOLO tahmini

```powershell
python predict.py C:\path\to\image.jpg --model b --device auto
```

## OOD'li tahmin

```powershell
python predict.py C:\path\to\image.jpg --model b --ood --device cpu
```

OOD çıktıları `outputs/phototrap_ood/` altında oluşturulur.

## OOD'siz tahmin

```powershell
python predict.py C:\path\to\image.jpg --model b --device cpu
```

## Testler

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m unittest src.phototrap_ood.validation.test_routing
```

Bu paket Model A dosyalarına import veya yol bağımlılığı içermez.
