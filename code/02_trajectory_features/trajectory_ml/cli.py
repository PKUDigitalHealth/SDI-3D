from __future__ import annotations

import argparse
import logging
import time

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe whole-night trajectory experiments."
    )
    parser.add_argument("--config", required=True, help="Path to analysis YAML.")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    started = time.time()
    config = load_config(arguments.config)
    logging.info("Starting trajectory experiment from %s", config["_config_path"])
    run_pipeline(config)
    logging.info("Finished in %.1f minutes", (time.time() - started) / 60.0)


if __name__ == "__main__":
    main()
