import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEED = 42
FRAC_TRAIN, FRAC_VAL = 0.70, 0.15      # test = remaining 0.15

def make_trace_split(data):
    """Split trace IDs (not rows), separately for each mobility class."""
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    for mob, g in data.groupby("mobility"):
        traces = np.array(sorted(g["trace_id"].unique()))
        rng.shuffle(traces)
        n = len(traces)
        n_tr = round(FRAC_TRAIN * n)
        n_va = round(FRAC_VAL * n)
        split["train"] += list(traces[:n_tr])
        split["val"] += list(traces[n_tr:n_tr + n_va])
        split["test"] += list(traces[n_tr + n_va:])
    return {k: sorted(v) for k, v in split.items()}

def fit_scaler(data, split, feature_cols):
    """Scaler sees TRAIN rows only."""
    train_rows = data[data["trace_id"].isin(split["train"])]
    return StandardScaler().fit(train_rows[feature_cols].to_numpy(dtype=np.float64))

def get_data(W, h, split, scaler):
    """Load windows for (W, h), split by trace, scale X. y stays in raw CQI units."""
    z = np.load(f"windows_W{W}_h{h}.npz")
    X, y = z["X"], z["y"]
    meta = pd.read_pickle(f"windows_W{W}_h{h}_meta.pkl")

    n, w, f = X.shape
    X = scaler.transform(X.reshape(-1, f)).reshape(n, w, f).astype(np.float32)

    out = {}
    for name in ["train", "val", "test"]:
        mask = meta["trace_id"].isin(split[name]).to_numpy()
        out[name] = (X[mask], y[mask], meta[mask].reset_index(drop=True))
    return out

if __name__ == "__main__":
    from make_features import BASE_FEATURES
    feature_cols = BASE_FEATURES + ["dCQI", "dSNR", "CQI_rmean5", "CQI_rstd5",
                                    "SNR_rmean5", "SNR_rstd5"]
    data = pd.read_pickle("features.pkl")

    split = make_trace_split(data)
    with open("split.json", "w") as f:
        json.dump(split, f, indent=1)

    scaler = fit_scaler(data, split, feature_cols)
    joblib.dump(scaler, "scaler.pkl")

    # --- sanity checks ---
    sets = [set(split[k]) for k in ["train", "val", "test"]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    print("No trace appears in more than one split - OK")

    tmob = data.groupby("trace_id")["mobility"].first()
    for name in ["train", "val", "test"]:
        mob = tmob[split[name]].value_counts().to_dict()
        print(f"{name:5s}: {len(split[name])} traces  {mob}")

    d = get_data(10, 3, split, scaler)
    for name, (X, y, m) in d.items():
        print(f"{name:5s}: X{X.shape}  windows {len(y)}  "
              f"Driving {int((m.mobility=='Driving').sum())} "
              f"Static {int((m.mobility=='Static').sum())}")