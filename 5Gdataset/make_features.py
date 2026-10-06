import pandas as pd

BASE_FEATURES = ["CQI", "SNR", "RSRP", "RSRQ", "Speed"]
HORIZONS = [1, 3, 5]
ROLL_WIN = 5

def make_features(d):
    d = d.sort_values(["seg_uid", "Timestamp"]).reset_index(drop=True)
    g = d.groupby("seg_uid")          # everything below stays inside a segment

    # difference features
    d["dCQI"] = g["CQI"].diff()
    d["dSNR"] = g["SNR"].diff()

    # rolling mean / std over the last 5 samples (current row included)
    for col in ["CQI", "SNR"]:
        r = g[col].rolling(ROLL_WIN, min_periods=ROLL_WIN)
        d[f"{col}_rmean{ROLL_WIN}"] = r.mean().reset_index(level=0, drop=True)
        d[f"{col}_rstd{ROLL_WIN}"] = r.std().reset_index(level=0, drop=True)

    # targets: CQI at t+h
    for h in HORIZONS:
        d[f"target_h{h}"] = g["CQI"].shift(-h)

    eng = ["dCQI", "dSNR", f"CQI_rmean{ROLL_WIN}", f"CQI_rstd{ROLL_WIN}",
           f"SNR_rmean{ROLL_WIN}", f"SNR_rstd{ROLL_WIN}"]
    feature_cols = BASE_FEATURES + eng

    # drop first rows of each segment where diff/rolling are undefined
    d = d.dropna(subset=feature_cols).reset_index(drop=True)
    return d, feature_cols

if __name__ == "__main__":
    cleaned = pd.read_pickle("cleaned.pkl")
    data, feature_cols = make_features(cleaned)
    print("Rows:", len(data), "| segments:", data["seg_uid"].nunique())
    print("Features:", feature_cols)
    for h in HORIZONS:
        print(f"target_h{h}: {data[f'target_h{h}'].notna().sum()} valid rows")
    data.to_pickle("features.pkl")