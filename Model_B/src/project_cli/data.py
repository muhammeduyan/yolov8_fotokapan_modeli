from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import IMAGE_EXTENSIONS, image_files, load_config, project_path, sha256_file, write_json


def _split_files(dataset: Path, split: str) -> list[Path]:
    return image_files(dataset / "images" / split)


def _label_for(image: Path, dataset: Path, split: str) -> Path:
    return dataset / "labels" / split / f"{image.stem}.txt"


def validate_dataset(dataset: Path, names: dict[str, str]) -> dict[str, Any]:
    names = {str(key): value for key, value in names.items()}
    report: dict[str, Any] = {"dataset": str(dataset), "splits": {}, "duplicates": [], "cross_dataset": []}
    seen: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        images = _split_files(dataset, split)
        classes: Counter[str] = Counter()
        boxes = 0
        empty = 0
        missing_labels: list[str] = []
        bad_labels: list[str] = []
        for image in images:
            label = _label_for(image, dataset, split)
            if not label.exists():
                missing_labels.append(str(image.relative_to(dataset)))
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                valid = len(fields) == 5
                if valid:
                    try:
                        class_id = str(int(fields[0]))
                        values = [float(value) for value in fields[1:]]
                        valid = class_id in names and all(0 <= value <= 1 for value in values)
                    except ValueError:
                        valid = False
                if not valid:
                    bad_labels.append(f"{label.relative_to(dataset)}:{line_number}")
                else:
                    classes[class_id] += 1
                    boxes += 1
            digest = sha256_file(image)
            seen[digest].append({"split": split, "image": str(image.relative_to(dataset))})
        report["splits"][split] = {
            "images": len(images), "boxes": boxes, "empty_images": empty,
            "classes": dict(sorted(classes.items())), "missing_labels": missing_labels,
            "bad_labels": bad_labels,
        }
    report["duplicates"] = [items for items in seen.values() if len({item["split"] for item in items}) > 1]
    report["valid"] = not report["duplicates"] and all(
        not values["missing_labels"] and not values["bad_labels"] for values in report["splits"].values()
    )
    return report


def clean_copy(source: Path, destination: Path, names: dict[str, str]) -> dict[str, Any]:
    names = {str(key): value for key, value in names.items()}
    if destination.exists():
        raise FileExistsError(f"Temiz veri seti zaten var: {destination}")
    destination.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)
    kept_hashes: dict[str, str] = {}
    removed: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for image in _split_files(source, split):
            digest = sha256_file(image)
            if digest in kept_hashes:
                removed.append({"split": split, "image": str(image.relative_to(source)), "duplicate_of": kept_hashes[digest], "sha256": digest})
                continue
            target_image = destination / "images" / split / image.name
            target_label = destination / "labels" / split / f"{image.stem}.txt"
            shutil.copy2(image, target_image)
            source_label = _label_for(image, source, split)
            if source_label.exists():
                shutil.copy2(source_label, target_label)
            kept_hashes[digest] = f"{split}/{image.name}"
    (destination / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "\n".join(f"  {key}: {value}" for key, value in names.items()) + "\n",
        encoding="utf-8",
    )
    report = validate_dataset(destination, names)
    report["source"] = str(source)
    report["removed"] = removed
    write_json(destination / "integrity_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Model B dataset and optionally create a non-destructive clean copy.")
    parser.add_argument("--model", choices=("b",), default="b")
    parser.add_argument("--clean-model-b", action="store_true", help="Create data/prepared/v0.3_multispecies_demo_clean_v2.")
    args = parser.parse_args(argv)
    configs = {"b": "configs/model_b.yaml"}
    selected = (args.model,)
    for key in selected:
        config = load_config(configs[key])
        dataset = project_path(config["data"]).parent
        report = validate_dataset(dataset, config["classes"])
        print(f"{key.upper()}: {dataset}")
        print(f"  valid={report['valid']} duplicates={len(report['duplicates'])}")
        for split, values in report["splits"].items():
            print(f"  {split}: images={values['images']} boxes={values['boxes']} empty={values['empty_images']}")
        if report["duplicates"]:
            for group in report["duplicates"]:
                print("  duplicate:", "; ".join(item["split"] + "/" + item["image"] for item in group))
        write_json(project_path("results/experiments") / f"dataset_{key}_audit.json", report)
        if key == "b" and args.clean_model_b:
            destination = project_path("data/prepared/v0.3_multispecies_demo_clean_v2")
            clean = clean_copy(dataset, destination, config["classes"])
            print(f"  clean_copy={destination} removed={len(clean['removed'])} valid={clean['valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
