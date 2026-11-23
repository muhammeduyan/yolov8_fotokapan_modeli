#!/usr/bin/env python3
"""
train.py: Local Wild Boar v0.4 (YOLOv8n) eğitim / ince ayar (fine-tuning) betiği.

Örnek Kullanım:
    python src/train.py --data ../v0.3_wild_boar_closeup/data.yaml
    python src/train.py --data ../v0.3_wild_boar_closeup/data.yaml --model yolov8n.pt --epochs 60 --freeze 0
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_existing_path(path_str: str) -> Path:
    """Göreli dosya yolunu önce çalışma dizininde, bulunamazsa Model_A kökünde çözümler."""
    raw = Path(path_str).expanduser()
    if raw.is_absolute() or raw.exists():
        return raw.resolve()
    project_candidate = PROJECT_ROOT / raw
    if project_candidate.exists():
        return project_candidate.resolve()
    return raw.resolve()


def select_device(requested: str | None) -> str:
    """Kullanıcı isteğine veya donanım uygunluğuna göre eğitim cihazını (cuda/mps/cpu) seçer."""
    if requested and requested.strip():
        return requested.strip()
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    """Eğitim argümanlarını doğrular, veri setini yükler ve YOLOv8n eğitimini başlatır."""
    parser = argparse.ArgumentParser(
        description="Local Wild Boar v0.4 (YOLOv8n) eğitim / ince ayar betiği"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "model" / "best.pt"),
        help="Başlangıç ağırlığı (fine-tune için mevcut best.pt, sıfırdan eğitim için yolov8n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT.parent / "v0.3_wild_boar_closeup" / "data.yaml"),
        help="YOLO data.yaml dosya yolu",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Epoch sayısı (fine-tune: 8, sıfırdan: 60)")
    parser.add_argument("--batch", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--imgsz", type=int, default=640, help="Girdi görsel boyutu")
    parser.add_argument("--freeze", type=int, default=10, help="Dondurulacak backbone katman sayısı")
    parser.add_argument("--lr0", type=float, default=0.0004, help="Başlangıç öğrenme oranı")
    parser.add_argument("--device", type=str, default=None, help="Cihaz (mps / 0 / cpu; boşsa otomatik)")
    parser.add_argument("--name", type=str, default="v0.4_bg_finetune", help="Run klasör adı")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs değeri 1 veya daha büyük olmalıdır.")
    if args.batch <= 0:
        parser.error("--batch değeri 1 veya daha büyük olmalıdır.")
    if args.imgsz <= 0:
        parser.error("--imgsz değeri 1 veya daha büyük olmalıdır.")
    if args.freeze < 0:
        parser.error("--freeze değeri 0 veya daha büyük olmalıdır.")
    if args.lr0 <= 0:
        parser.error("--lr0 değeri 0'dan büyük olmalıdır.")
    if not args.name or not args.name.strip():
        parser.error("--name boş olamaz.")

    data_path = _resolve_existing_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Veri seti config dosyası bulunamadı: {data_path}\n"
            "Harici v0.3_wild_boar_closeup veri setini indirip --data parametresiyle yolunu verin."
        )

    model_arg = Path(args.model).expanduser()
    if not model_arg.is_absolute() and not model_arg.exists():
        project_model = PROJECT_ROOT / model_arg
        model_path = str(project_model) if project_model.exists() else args.model
    else:
        model_path = str(model_arg)

    device = select_device(args.device)
    print(f"Cihaz: {device} | Model: {model_path} | Veri: {data_path}")

    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        freeze=args.freeze if args.freeze > 0 else None,
        lr0=args.lr0,
        close_mosaic=min(8, args.epochs),
        project=str(PROJECT_ROOT / "runs" / "wild_boar"),
        name=args.name.strip(),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
