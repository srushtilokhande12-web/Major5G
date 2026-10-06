import json
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from split_data import get_data
from baselines import raw_cqi_window

SEED = 42
WINDOWS = [5, 10, 20]
DEPTHS = [3, 5, 7]
HORIZONS = [1, 3, 5]
MAX_TREES = 600

def prep(d, name, scaler):
    X, y, _ = d[name]
    base = raw_cqi_window(X, scaler)[:, -1]       # CQI(t) in real units
    return X.reshape(len(X), -1), y, base

def make_model(depth, n_estimators, early_stop=None):
    return XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=depth,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:absoluteerror", eval_metric="mae",
        tree_method="hist", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=early_stop)

if __name__ == "__main__":
    split = json.load(open("split.json"))
    scaler = joblib.load("scaler.pkl")
    os.makedirs("models", exist_ok=True)

    # ---------- 1. tune on VAL ----------
    rows = []
    for h in HORIZONS:
        for W in WINDOWS:
            d = get_data(W, h, split, scaler)
            Xtr, ytr, btr = prep(d, "train", scaler)
            Xva, yva, bva = prep(d, "val", scaler)
            persist = np.abs(bva - yva).mean()
            for depth in DEPTHS:
                m = make_model(depth, MAX_TREES, early_stop=30)
                m.fit(Xtr, ytr - btr, eval_set=[(Xva, yva - bva)], verbose=False)
                pred = np.clip(m.predict(Xva) + bva, 0, 15)
                mae = np.abs(pred - yva).mean()
                rows.append({"h": h, "W": W, "max_depth": depth,
                             "n_estimators": m.best_iteration + 1,
                             "val_MAE": mae, "val_persistence_MAE": persist})
                print(f"h={h} W={W:2d} depth={depth}  trees={m.best_iteration+1:3d}  "
                      f"val MAE={mae:.4f}  (persistence {persist:.4f})")

    res = pd.DataFrame(rows)
    res.to_csv("tuning_results_xgb.csv", index=False)

    # ---------- 2. pick one final W, then best depth per horizon ----------
    best_per = res.loc[res.groupby(["h", "W"])["val_MAE"].idxmin()]
    best_W = int(best_per.groupby("W")["val_MAE"].mean().idxmin())
    print(f"\nFinal window size W = {best_W}")

    config = {"W": best_W}
    for h in HORIZONS:
        r = best_per[(best_per.h == h) & (best_per.W == best_W)].iloc[0]
        config[str(h)] = {"max_depth": int(r.max_depth),
                          "n_estimators": int(r.n_estimators)}

    # ---------- 3. refit on TRAIN with the chosen settings and save ----------
    for h in HORIZONS:
        c = config[str(h)]
        d = get_data(best_W, h, split, scaler)
        Xtr, ytr, btr = prep(d, "train", scaler)
        Xva, yva, bva = prep(d, "val", scaler)
        m = make_model(c["max_depth"], c["n_estimators"])
        m.fit(Xtr, ytr - btr, verbose=False)
        pred = np.clip(m.predict(Xva) + bva, 0, 15)
        print(f"h={h}: {c}  val MAE={np.abs(pred - yva).mean():.4f}")
        joblib.dump(m, f"models/xgb_h{h}.joblib")

    json.dump(config, open("best_config.json", "w"), indent=1)