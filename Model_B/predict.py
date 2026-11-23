from __future__ import annotations

import argparse
from pathlib import Path

from src.project_cli.common import device_value, load_config, project_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict one image or a folder with Model B.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", choices=("b",), default="b")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/predictions"))
    parser.add_argument("--ood", action="store_true", help="Use the existing active YOLO+OOD application; single-image only.")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.ood:
        if source.is_dir():
            raise ValueError("--ood yalnızca tek görüntü için kullanılabilir.")
        from src.phototrap_ood.runtime import PhotoRuntime
        result = PhotoRuntime(args.device).predict(source, args.output)
        print(result["message"])
        print(result["outputs"])
        return 0
    from ultralytics import YOLO
    config = load_config(f"configs/model_{args.model}.yaml")
    model = YOLO(str(project_path(config["weights"])))
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_name = config["name"]
    suffix = 2
    while (output / run_name).exists():
        run_name = f"{config['name']}_{suffix}"
        suffix += 1
    results = model.predict(source=str(source), conf=args.conf, imgsz=config["train"]["imgsz"], max_det=300,
                            device=device_value(args.device), save=True, project=str(output), name=run_name,
                            exist_ok=False, verbose=False)
    print(f"predictions={len(results)} output={output / run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
