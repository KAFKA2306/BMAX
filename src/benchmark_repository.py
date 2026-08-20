"""Repository-level loader for benchmark contract, market evidence, and scenarios."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark_contract import load_repository_dataset


def load_benchmark_repository(dataset_path: Path) -> dict[str, Any]:
    dataset = load_repository_dataset(dataset_path)
    existing = list(dataset.get("scenarios", []))
    scenario_dir = dataset_path.parent / "scenarios"
    external = []
    if scenario_dir.exists():
        for path in sorted(scenario_dir.glob("*.json")):
            external.append(json.loads(path.read_text(encoding="utf-8")))
    dataset["scenarios"] = [*existing, *external]
    return dataset
