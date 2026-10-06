import json
import os
import joblib
import numpy as np
import pandas as pd
from split_data import get_data
from baselines import raw_cqi_window, metrics

SEED = 42
HORIZONS = [1, 3, 5]
N_BOOT = 2000
MOBS = ["All", "Driving", "Static"]
rng = np.random.default_rng(SEED)

def predict_xgb(h, X, base):
    m = joblib.load(f"models/xgb_h{h}.joblib")
    return np.clip(m.predict(X.reshape(len(X), -1)) + base, 0, 15)

def predict_gru(h, X, base):
    import torch
    from train_gru import GRUNet
    net = GRUNet(X.shape[2])
    net.load_state_dict(torch.load(f"models/gru_h{h}.pt"))
    net.eval()
    with torch.no_grad():
        r = net(torch.tensor(X)).numpy()
    return np.clip(r + base, 0, 15)

def trace_bootstrap(trace_ids, y, p_base, p_model):
    """MAE gain of model over persistence, with a 95% CI from resampling traces."""
    df = pd.DataFrame({"t": trace_ids,
                       "ep": np.abs(p_base - y),
                       "em": np.abs(p_model - y)})
    g = df.groupby("t").agg(sp=("ep", "sum"), sm=("em", "sum"), n=("ep", "size"))
    sp, sm, n = g.sp.to_numpy(), g.sm.to_numpy(), g.n.to_numpy()
    k = len(g)
    gain = (sp.sum() - sm.sum()) / n.sum()
    idx = rng.integers(0, k, size=(N_BOOT, k))
    boot = (sp[idx].sum(1) - sm[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return gain, lo, hi, int((sm < sp).sum()), k

if __name__ == "__main__":
    split = json.load(open("split.json"))
    scaler = joblib.load("scaler.pkl")
    W = json.load(open("best_config.json"))["W"]
    print(f"Evaluating on TEST traces only, W={W}\n")

    rows, gain_rows, pred_frames = [], [], []
    for h in HORIZONS:
        X, y, meta = get_data(W, h, split, scaler)["test"]
        base = raw_cqi_window(X, scaler)[:, -1]
        preds = {"Persistence": base, "XGBoost": predict_xgb(h, X, base)}
        if os.path.exists(f"models/gru_h{h}.pt"):
            preds["GRU"] = predict_gru(h, X, base)

        for mob in MOBS:
            mask = (np.ones(len(y), bool) if mob == "All"
                    else (meta["mobility"] == mob).to_numpy())
            for name, p in preds.items():
                rows.append({"model": name, "h": h, "mobility": mob,
                             **metrics(y[mask], p[mask])})
                if name != "Persistence":
                    gain, lo, hi, wins, k = trace_bootstrap(
                        meta["trace_id"].to_numpy()[mask], y[mask],
                        base[mask], p[mask])
                    gain_rows.append({"model": name, "h": h, "mobility": mob,
                                      "MAE_gain": gain, "CI_low": lo, "CI_high": hi,
                                      "traces_better": f"{wins}/{k}"})

        out = meta.copy()
        out["h"] = h
        out["y_true"] = y
        for name, p in preds.items():
            out[f"pred_{name}"] = p
        pred_frames.append(out)

    res = pd.DataFrame(rows)
    gains = pd.DataFrame(gain_rows)
    res.to_csv("results_test.csv", index=False)
    gains.to_csv("results_test_gains.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv("test_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    for col in ["MAE", "RMSE", "within1"]:
        print(f"=== TEST : {col} ===")
        print(res.pivot_table(index=["mobility", "model"], columns="h",
                              values=col).round(3), "\n")
    print("=== TEST : MAE gain over persistence (positive = better) ===")
    print(gains.round(3).to_string(index=False))
    print("\nTest windows per group:")
    print(res[res.model == "Persistence"].pivot_table(
        index="mobility", columns="h", values="n").astype(int))