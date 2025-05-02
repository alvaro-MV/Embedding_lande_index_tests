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
        serie = ast.literal_eval(raw_str)

        # Validate it's a numeric list
        if isinstance(serie, list) and len(serie) >= 3:
            serie = np.array(serie, dtype=float)

            # Check for variability
            if np.std(serie) > VARIABILITY_THRESHOLD:
                scaled = zscore(serie)
                result = mk.original_test(scaled)
                results.append({
                    "row": idx,
                    "trend": result.trend,
                    "p_value": result.p,
                    "tau": result.Tau
                })
            else:
                results.append({
                    "row": idx,
                    "trend": "Too flat (after scaling)",
                    "p_value": None,
                    "tau": None
                })
        else:
            results.append({
                "row": idx,
                "trend": "Invalid or short series",
                "p_value": None,
                "tau": None
            })

    except Exception as e:
        results.append({
            "row": idx,
            "trend": f"Error: {str(e)}",
            "p_value": None,
            "tau": None
        })

# === Print results ===
for r in results:
    print(f"Row {r['row']}: trend={r['trend']}, p={r['p_value']}, tau={r['tau']}")
