"""使用 show_3D_white.py 批量重绘四个白底病例图。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("show_3D_white.py")
CSV_FILES = [
    Path(r"C:\Users\SaberShuo\Desktop\case_study\shhs1-200419_features_predictions.csv"),
    Path(r"C:\Users\SaberShuo\Desktop\case_study\shhs1-201674_features_predictions.csv"),
    Path(r"C:\Users\SaberShuo\Desktop\case_study\shhs1-205218_features_predictions.csv"),
    Path(r"C:\Users\SaberShuo\Desktop\case_study\shhs1-201575_features_predictions.csv"),
]


def main() -> None:
    missing = [str(path) for path in CSV_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    for csv_path in CSV_FILES:
        print(f"[render] {csv_path.name}", flush=True)
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["SHOW_3D_NO_SHOW"] = "1"
        env["SHOW_3D_CSV_PATH"] = str(csv_path)
        subprocess.run([sys.executable, str(SCRIPT)], env=env, check=True)

    print("[done] 四个白底病例图已全部生成。")


if __name__ == "__main__":
    main()
