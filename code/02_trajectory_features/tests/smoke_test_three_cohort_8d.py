from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "sleep_autoencoder_8d.py"


def write_cohort(folder: Path, prefix: str, count: int, offset: float) -> None:
    rng = np.random.default_rng(123 + int(offset * 10))
    folder.mkdir(parents=True)
    for index in range(count):
        length = 30 + index
        phase = np.linspace(0, 4 * np.pi, length)
        pd.DataFrame(
            {
                "wake_depth_score": np.sin(phase) + offset
                + rng.normal(0, 0.05, length),
                "rem_logit_rem": np.cos(phase / 2) - offset
                + rng.normal(0, 0.05, length),
            }
        ).to_csv(folder / f"{prefix}-{index:03d}.csv", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sleep_ae_three_cohort_") as temp:
        root = Path(temp)
        cfs = root / "cfs"
        mros = root / "mros"
        shhs = root / "shhs"
        output = root / "output"
        write_cohort(cfs, "cfs", 5, 0.0)
        write_cohort(mros, "mros", 5, 0.3)
        write_cohort(shhs, "shhs", 4, 0.8)
        specification = importlib.util.spec_from_file_location("ae8", SCRIPT)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        settings = module.Settings()
        settings.cfs_dir = cfs
        settings.mros_dir = mros
        settings.shhs_dir = shhs
        settings.output_dir = output
        settings.file_pattern = "*.csv"
        settings.epochs = 1
        settings.batch_size = 4
        settings.hidden = 8
        settings.cpu = True
        module.train(settings)
        expected = {
            "cfs_embeddings_8d.csv": 5,
            "mros_embeddings_8d.csv": 5,
            "shhs_external_embeddings_8d.csv": 4,
            "all_datasets_embeddings_8d.csv": 14,
        }
        feature_columns = [f"feature_{index}" for index in range(1, 9)]
        for filename, rows in expected.items():
            frame = pd.read_csv(output / filename)
            assert len(frame) == rows
            assert feature_columns == [
                column for column in frame if column.startswith("feature_")
            ]
        external = pd.read_csv(output / "shhs_external_embeddings_8d.csv")
        assert set(external["split"]) == {"external_test"}
        assert set(external["dataset"]) == {"SHHS"}
        print("three_cohort_8d_smoke_test_ok")


if __name__ == "__main__":
    main()
