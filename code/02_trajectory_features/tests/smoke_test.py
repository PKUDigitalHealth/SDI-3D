from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from trajectory_ml.config import load_config  # noqa: E402
from trajectory_ml.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="trajectory_ml_smoke_") as folder:
        root = Path(folder)
        trajectories = root / "trajectories"
        output = root / "output"
        trajectories.mkdir()
        harmonized_rows = []
        outcome_rows = []
        for index in range(16):
            subject_id = str(200000 + index)
            event = index % 2
            length = 120 + index
            time = np.arange(length)
            phase = np.linspace(0, 6 * np.pi, length)
            depth = np.sin(phase) + 0.5 * event + rng.normal(0, 0.2, length)
            rem_logit = np.cos(phase / 2) + 0.7 * event + rng.normal(0, 0.2, length)
            label = np.select(
                [rem_logit > 0.8, depth < -0.4, depth > 0.8],
                [4, 0, 3],
                default=2,
            )
            pd.DataFrame(
                {
                    "nsrrid": subject_id,
                    "visit": 1,
                    "window_center": time,
                    "wake_depth_score": depth,
                    "rem_logit_rem": rem_logit,
                    "label": label,
                }
            ).to_csv(
                trajectories / f"shhs1-{subject_id}_features_predictions.csv",
                index=False,
            )
            harmonized_rows.append(
                {
                    "nsrrid": subject_id,
                    "age": 50 + index,
                    "bmi": 24 + index / 10,
                    "sex": "F" if index % 2 else "M",
                }
            )
            outcome_rows.append(
                {
                    "nsrrid": subject_id,
                    "incident_cvd": event,
                    "followup_years": 3 if event else 7,
                    "prevalent_cvd": 0,
                }
            )
        harmonized_path = root / "harmonized.csv"
        outcome_path = root / "outcomes.csv"
        pd.DataFrame(harmonized_rows).to_csv(harmonized_path, index=False)
        pd.DataFrame(outcome_rows).to_csv(outcome_path, index=False)

        with (PROJECT / "configs" / "shhs_cvd.example.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            config = yaml.safe_load(handle)
        config["paths"] = {
            "trajectory_dir": str(trajectories),
            "harmonized_file": str(harmonized_path),
            "outcomes_file": str(outcome_path),
            "output_dir": str(output),
        }
        config["data"]["minimum_epochs"] = 60
        config["metadata"]["categorical_covariates"] = ["sex"]
        config["encoder"].update(
            {
                "architecture": "transformer",
                "embedding_dim": 8,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "batch_size": 8,
                "epochs": 2,
                "patience": 2,
            }
        )
        config["validation"].update(
            {
                "outer_splits": 2,
                "inner_splits": 2,
                "tuning_iterations": 1,
                "bootstrap_iterations": 30,
                "n_jobs": 1,
            }
        )
        config["gmm"].update(
            {"minimum_components": 2, "maximum_components": 3, "repetitions": 2}
        )
        config_path = root / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        run_pipeline(load_config(config_path))

        required_outputs = [
            "oof_predictions.csv",
            "oof_summary_metrics.csv",
            "bootstrap_auc_comparisons.csv",
            "final_embeddings.csv",
            "subtype_assignments.csv",
            "run_summary.json",
        ]
        missing = [name for name in required_outputs if not (output / name).exists()]
        if missing:
            raise AssertionError(f"Missing outputs: {missing}")
        predictions = pd.read_csv(output / "oof_predictions.csv")
        probability_columns = [
            column for column in predictions if column.startswith("probability_")
        ]
        if predictions[probability_columns].isna().any().any():
            raise AssertionError("OOF predictions contain missing values.")
        print("smoke_test_ok", len(predictions), sorted(required_outputs))


if __name__ == "__main__":
    main()
