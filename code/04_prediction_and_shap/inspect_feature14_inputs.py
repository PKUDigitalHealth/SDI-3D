from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14")
FILES = [
    "cfs_SDI_14_Features.csv",
    "cfs_embeddings_8d.csv",
    "mros_SDI_14_Features.csv",
    "mros_embeddings_8d.csv",
    "shhs_SDI_14_Features.csv",
    "shhs_external_embeddings_8d.csv",
]


for name in FILES:
    path = ROOT / name
    data = pd.read_csv(path)
    print(f"FILE {name} shape={data.shape}")
    print("COLS", list(data.columns))
    print(data.head(2).to_string(index=False))
    print("DTYPES", data.dtypes.astype(str).to_dict())
    print("MISSING", data.isna().sum()[data.isna().sum() > 0].to_dict())
    print()
