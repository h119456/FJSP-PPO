"""Create a zero-auxiliary-cost copy of the small dataset without modifying it."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "dataset"
TARGET = ROOT / "data" / "dataset1_cost0"
SPLITS = ("train", "validation", "test")


def main() -> None:
    manifest = {
        "derived_from": "data/dataset",
        "changes": {
            "stress_relief.cost": 0.0,
            "compensation.cost": 0.0,
        },
        "unchanged": [
            "processing durations",
            "stress-relief and compensation durations",
            "deformation parameters",
            "compensation threshold",
            "objective weights",
        ],
        "splits": {},
    }
    for split in SPLITS:
        source_files = sorted((SOURCE / split).glob("*.json"))
        target_dir = TARGET / split
        target_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for source_path in source_files:
            data = json.loads(source_path.read_text(encoding="utf-8"))
            data["name"] = f"{data['name']}_cost0"
            data["stress_relief"]["cost"] = 0.0
            data["compensation"]["cost"] = 0.0
            target_path = target_dir / source_path.name
            target_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(f"{split}/{target_path.name}")
        manifest["splits"][split] = written

    (TARGET / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
