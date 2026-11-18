from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")

    required_sections = [
        "paths",
        "data",
        "metadata",
        "outcome",
        "scaling",
        "handcrafted",
        "encoder",
        "validation",
        "xgboost",
        "gmm",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError(f"Missing configuration sections: {', '.join(missing)}")

    resolved = deepcopy(config)
    for key in [
        "trajectory_dir",
        "harmonized_file",
        "outcomes_file",
        "output_dir",
    ]:
        value = Path(resolved["paths"][key]).expanduser()
        if not value.is_absolute():
            value = (config_path.parent / value).resolve()
        resolved["paths"][key] = str(value)
    resolved["_config_path"] = str(config_path)
    return resolved
