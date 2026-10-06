import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from split_data import get_data

W = 10
HORIZONS = [1, 3, 5]
CQI_IDX = 0                       # CQI is the first feature column

def raw_cqi_window(X, scaler):
    """Undo scaling for the CQI channel -> (N, W) in real CQI units."""
    return X[:, :, CQI_IDX] * scaler.scale_[CQI_IDX] + scaler.mean_[CQI_IDX]

def metrics(y, pred):
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    pr = np.clip(np.rint(pred), 0, 15)
    return {"MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "within1": (np.abs(pr - y) <= 1).mean(),
            "n": len(y)}

def evaluate(name, h, split_name, y, pred, meta, rows):
    for mob in ["All", "Driving", "Static"]:
        m = np.ones(len(y), bool) if mob == "All" else (meta["mobility"] == mob).to_numpy()
        rows.append({"model": name, "h": h, "set": split_name, "mobility": mob,
                     **metrics(y[m], pred[m])})

if __name__ == "__main__":
    split = json.load(open("split.json"))
    scaler = joblib.load("scaler.pkl")
    rows = []

    for h in HORIZONS:
        d = get_data(W, h, split, scaler)
        Xtr, ytr, _ = d["train"]
        lr = LinearRegression().fit(Xtr.reshape(len(Xtr), -1), ytr)

        for sname in ["val", "test"]:
            X, y, meta = d[sname]
            cq = raw_cqi_window(X, scaler)
            preds = {
                "Persistence":   cq[:, -1],
                "MovingAvg3":    cq[:, -3:].mean(axis=1),
                "MovingAvg5":    cq[:, -5:].mean(axis=1),
                "LinearReg":     np.clip(lr.predict(X.reshape(len(X), -1)), 0, 15),
            }
            for name, p in preds.items():
                evaluate(name, h, sname, y, p, meta, rows)

            if sname == "test":
                out = meta.copy()
                out["y_true"] = y
                for name, p in preds.items():
                    out[f"pred_{name}"] = p
                out.to_pickle(f"test_preds_baselines_h{h}.pkl")

    res = pd.DataFrame(rows)
    res.to_csv("results_baselines.csv", index=False)

    pd.set_option("display.width", 200)
    for sname in ["val", "test"]:
        t = res[(res.set == sname)]
        print(f"\n=== {sname.upper()} : MAE ===")
        print(t.pivot_table(index=["mobility", "model"], columns="h", values="MAE").round(3))
    t = res[(res.set == "test")]
    print("\n=== TEST : within ±1 CQI ===")
    print(t.pivot_table(index=["mobility", "model"], columns="h", values="within1").round(3))