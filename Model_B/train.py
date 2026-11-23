from __future__ import annotations

import argparse

from src.project_cli.common import device_value, load_config, project_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Model B without overwriting an existing run.")
    parser.add_argument("--model", choices=("b",), default="b")
    parser.add_argument("--data", help="Optional data.yaml override.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", help="New run name; required to avoid accidental overwrite.")
    args = parser.parse_args(argv)
    config = load_config(f"configs/model_{args.model}.yaml")
    data = project_path(args.data or config["data"])
    name = args.name or f"{config['name']}_new"
    run_root = project_path("runs/experiments")
    target = run_root / name
    if target.exists():
        raise FileExistsError(f"Run already exists; choose --name: {target}")
    from ultralytics import YOLO
    model = YOLO(str(project_path(config.get("base_model", "yolov8n.pt"))))
    train = config["train"]
    result = model.train(data=str(data), epochs=args.epochs or train["epochs"], batch=train["batch"], imgsz=train["imgsz"],
                         seed=train["seed"], deterministic=train["deterministic"], device=device_value(args.device),
                         workers=train["workers"], pretrained=train["pretrained"], optimizer=train["optimizer"],
                         lr0=train["lr0"], project=str(run_root), name=name, exist_ok=False, save=True, plots=True)
    print(f"run={result.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
