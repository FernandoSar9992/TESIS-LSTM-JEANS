import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Cronograma_Semanas_2023_2026_v24.xlsx"

xls = pd.ExcelFile(SOURCE)
report = {"sheets": xls.sheet_names, "detail": {}}
for sheet in xls.sheet_names:
    frame = pd.read_excel(SOURCE, sheet_name=sheet)
    report["detail"][sheet] = {
        "shape": list(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "head": frame.head(8).replace({np.nan: None}).to_dict(orient="records"),
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
    }

print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
