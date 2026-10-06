import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZONS = [1, 3, 5]
WINDOW = 10

def check_contiguous(data):
    """Every segment must advance exactly 1 s per row."""
    dt = data.groupby("seg_uid")["Timestamp"].diff().dropna()
    bad = (dt != pd.Timedelta("1s")).sum()
    assert bad == 0, f"{bad} non-1s steps inside segments"

def make_windows(data, feature_cols, W, h):
    target_col = f"target_h{h}"
    X_list, y_list, meta_list = [], [], []

    for seg_uid, seg in data.groupby("seg_uid", sort=False):
        if len(seg) < W + h:
            continue
        feats = seg[feature_cols].to_numpy(dtype=np.float32)
        tgt = seg[target_col].to_numpy()

        # windows end at rows W-1 ... T-1
        win = sliding_window_view(feats, (W, feats.shape[1]))[:, 0]   # (T-W+1, W, F)
        end_idx = np.arange(W - 1, len(seg))
        y = tgt[end_idx]
        ok = ~np.isnan(y)                                             # drop last h rows

        X_list.append(win[ok])
        y_list.append(y[ok])
        m = seg.iloc[end_idx[ok]][["trace_id", "seg_uid", "mobility", "app", "Timestamp"]]
        m = m.assign(cqi_now=seg["CQI"].to_numpy()[end_idx[ok]])
        meta_list.append(m)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list).astype(np.float32)
    meta = pd.concat(meta_list, ignore_index=True)
    return X, y, meta

if __name__ == "__main__":
    from make_features import BASE_FEATURES
    data = pd.read_pickle("features.pkl")
    feature_cols = BASE_FEATURES + ["dCQI", "dSNR", "CQI_rmean5", "CQI_rstd5",
                                    "SNR_rmean5", "SNR_rstd5"]
    check_contiguous(data)
    print("Segments are contiguous (1 s steps) - OK")

    for W in [5, 10, 20]:
        for h in HORIZONS:
            X, y, meta = make_windows(data, feature_cols, W, h)
            print(f"W={W:2d} h={h}: X{X.shape} y{y.shape}")
            if W == WINDOW:
                np.savez_compressed(f"windows_W{W}_h{h}.npz", X=X, y=y)
                meta.to_pickle(f"windows_W{W}_h{h}_meta.pkl")