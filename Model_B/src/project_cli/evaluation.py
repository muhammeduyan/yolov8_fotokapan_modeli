from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import device_value, load_config, project_path, sha256_file, write_json


def _empty_analysis(model: Any, data_yaml: Path, evaluation: dict[str, Any], device: str) -> dict[str, Any]:
    dataset = data_yaml.parent
    image_root = dataset / "images" / evaluation.get("split", "test")
    label_root = dataset / "labels" / evaluation.get("split", "test")
    empty_images = []
    for image in sorted(image_root.glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        label = label_root / f"{image.stem}.txt"
        if label.exists() and not label.read_text(encoding="utf-8-sig").strip():
            empty_images.append(image)
    false_images = 0
    false_boxes = 0
    for image in empty_images:
        result = model.predict(source=str(image), conf=evaluation.get("conf", 0.25),
                               iou=evaluation.get("iou", 0.50), imgsz=evaluation.get("imgsz", 640),
                               max_det=evaluation.get("max_det", 300), device=device_value(device), verbose=False,
                               save=False)[0]
        count = len(result.boxes) if result.boxes is not None else 0
        false_boxes += count
        false_images += int(count > 0)
    total = len(empty_images)
    return {"empty_images": total, "images_with_false_positive": false_images,
            "false_positive_boxes": false_boxes,
            "correct_rejection_rate": (1 - false_images / total) if total else None}


def _metric(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_metric(item) for item in value]
    return value


def evaluate(config_path: str, data_override: str | None = None, output: str | None = None, device: str = "auto") -> Path:
    config = load_config(config_path)
    from ultralytics import YOLO

    weights = project_path(config["weights"])
    data_argument = data_override or config["data"]
    data = project_path(data_argument)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)
    evaluation = config["evaluation"]
    model = YOLO(str(weights))
    output_root = project_path(output or config.get("output_root", "results/experiments"))
    run_name = config["name"]
    suffix = 2
    while (output_root / run_name).exists():
        run_name = f"{config['name']}_{suffix}"
        suffix += 1
    metrics = model.val(
        data=str(data_argument), split=evaluation.get("split", "test"), imgsz=config["train"]["imgsz"],
        conf=evaluation.get("conf", 0.25), iou=evaluation.get("iou", 0.50),
        max_det=evaluation.get("max_det", 300), device=device_value(device),
        workers=0, plots=True, project=str(output_root), name=run_name, exist_ok=False,
    )
    result: dict[str, Any] = {
        "schema_version": 1, "model": config["name"], "config": str(project_path(config_path)),
        "weights": str(weights), "weights_sha256": sha256_file(weights), "data": str(data),
        "split": evaluation.get("split", "test"), "parameters": evaluation,
        "metrics": {"precision": _metric(metrics.box.mp), "recall": _metric(metrics.box.mr),
                     "mAP50": _metric(metrics.box.map50), "mAP50-95": _metric(metrics.box.map)},
        "per_class": {"precision": _metric(getattr(metrics.box, "p", [])), "recall": _metric(getattr(metrics.box, "r", [])),
                      "mAP50": _metric(getattr(metrics.box, "ap50", [])), "mAP50-95": _metric(getattr(metrics.box, "ap", []))},
        "names": {str(key): value for key, value in model.names.items()},
        "empty_image_analysis": _empty_analysis(model, data, evaluation, device),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }
    result_path = Path(getattr(metrics, "save_dir", project_path(output or "results/experiments"))) / "metrics.json"
    write_json(result_path, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"results={result_path}")
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Model B on its declared test split.")
    parser.add_argument("--model", choices=("b",), default="b")
    parser.add_argument("--data", help="Optional data.yaml override; use a clean dataset explicitly.")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    models = (args.model,)
    for model in models:
        evaluate(f"configs/model_{model}.yaml", args.data if args.model != "both" else None, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
