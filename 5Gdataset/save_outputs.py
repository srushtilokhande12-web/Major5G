import glob
import os
import shutil
import joblib
import numpy as np
import pandas as pd

OUT = "outputs"
os.makedirs(f"{OUT}/models", exist_ok=True)

# 1. cleaned dataset -> parquet
d = pd.read_pickle("cleaned.pkl")
for c in d.select_dtypes("object").columns:      # avoid mixed-type errors
    d[c] = d[c].astype(str)
d.to_parquet(f"{OUT}/cleaned_5g.parquet", index=False)

# 2. models + everything needed to reuse them
for f in glob.glob("models/*"):
    shutil.copy(f, f"{OUT}/models/")
for f in ["scaler.pkl", "split.json", "best_config.json",
          "tuning_results_xgb.csv", "results_baselines.csv",
          "results_test.csv", "results_test_gains.csv"]:
    shutil.copy(f, OUT)

# 3. test predictions (adds an integer CQI column for link adaptation)
p = pd.read_csv("test_predictions.csv")
p["pred_XGBoost_cqi"] = np.clip(np.rint(p["pred_XGBoost"]), 0, 15).astype(int)
p.to_csv(f"{OUT}/test_predictions.csv", index=False)

# ---------- checks ----------
chk = pd.read_parquet(f"{OUT}/cleaned_5g.parquet")
print(f"Parquet: {len(chk)} rows | {chk['trace_id'].nunique()} traces | "
      f"{chk['seg_uid'].nunique()} segments")
assert len(chk) == len(d)

print("\nPredictions per horizon:")
for h, g in p.groupby("h"):
    print(f"h={h}: {len(g)} rows | persistence MAE "
          f"{(g.pred_Persistence - g.y_true).abs().mean():.3f} | "
          f"XGBoost MAE {(g.pred_XGBoost - g.y_true).abs().mean():.3f}")

print("\nModels:")
for h in [1, 3, 5]:
    m = joblib.load(f"{OUT}/models/xgb_h{h}.joblib")
    print(f"xgb_h{h}: loaded OK ({type(m).__name__})")

print("\nFiles in outputs/:")
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        path = os.path.join(root, f)
        print(f"{os.path.getsize(path)/1e6:8.2f} MB  {path}")