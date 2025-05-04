import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import zscore
import ast

# === Configuration ===
CSV_PATH = "data/tendencias_expositivo"
COLUMN_NAME = "lande_intra"
VARIABILITY_THRESHOLD = 1e-12

# === Load data ===
df = pd.read_csv(CSV_PATH)
# === Check column exists ===
if COLUMN_NAME not in df.columns:
    raise ValueError(f"Column '{COLUMN_NAME}' not found in CSV.")

# === Analyze each series ===
results = []
for idx, raw_str in enumerate(df[COLUMN_NAME]):
    try:
        # Convert string to list
        print(f"idx: {idx}   raw: {raw_str}\n")
        serie = np.fromstring(
            raw_str.strip("[]").replace("\n", " "),   # ← elimina '\n'
            sep=" "                                   # ← separador = espacio
        )
        result = mk.original_test(serie)
        results.append({
            "name": df.iloc[idx, :].loc['text_name'],
            "trend": result.trend,
            "p_value": result.p,
            "tau": result.Tau
        })

    except Exception as e:
        results.append({
            "name": df.iloc[idx, :].loc['text_name'],
            "trend": f"Error: {str(e)}",
            "p_value": None,
            "tau": None
        })

# === Print results ===
for r in results:
    print(f"{r['name']}: trend={r['trend']}, p={r['p_value']}, tau={r['tau']}\n")
