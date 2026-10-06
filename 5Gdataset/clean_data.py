import pandas as pd

NUM_COLS = ["RSRP", "RSRQ", "SNR", "CQI", "Speed", "DL_bitrate"]
MAX_FFILL = 3          # forward-fill gaps of up to 3 s
MIN_SEG_LEN = 30       # drop segments shorter than this (seconds)
KEEP_5G_ONLY = True

def clean(raw):
    d = raw.copy()
    print(f"Start: {len(d)} rows")

    # 1. numeric conversion ("-" -> NaN)
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    print("NaN after conversion:\n", d[NUM_COLS].isna().sum().to_string())

    # 2. drop duplicate timestamps within a trace (keep last)
    d = d.sort_values(["trace_id", "Timestamp"])
    before = len(d)
    d = d.drop_duplicates(["trace_id", "Timestamp"], keep="last")
    print(f"Removed {before - len(d)} duplicate-timestamp rows")

    # 4. 5G filter / flag
    d["is_5g"] = (d["NetworkMode"] == "5G").astype(int)
    print(f"5G share: {d['is_5g'].mean():.1%}")
    if KEEP_5G_ONLY:
        d = d[d["is_5g"] == 1]
        print(f"Kept 5G only: {len(d)} rows")

    # 3. resample each trace to a 1 s grid, ffill <=3 s, cut at longer gaps
    out = []
    for tid, g in d.groupby("trace_id"):
        g = g.set_index("Timestamp")
        full = pd.date_range(g.index.min(), g.index.max(), freq="1s")
        g = g.reindex(full)
        g["is_filled"] = g["CQI"].isna() | g["trace_id"].isna()
        g = g.ffill(limit=MAX_FFILL)
        g = g.dropna(subset=["trace_id"] + NUM_COLS)
        if g.empty:
            continue
        g.index.name = "Timestamp"
        g = g.reset_index()
        g["segment_id"] = (g["Timestamp"].diff() != pd.Timedelta("1s")).cumsum()
        out.append(g)

    d = pd.concat(out, ignore_index=True)
    d["seg_uid"] = d["trace_id"] + "_" + d["segment_id"].astype(str)

    seg_len = d.groupby("seg_uid")["Timestamp"].transform("size")
    d = d[seg_len >= MIN_SEG_LEN].reset_index(drop=True)

    d["CQI"] = d["CQI"].astype(int)
    print(f"Final: {len(d)} rows | {d['trace_id'].nunique()} traces | "
          f"{d['seg_uid'].nunique()} segments")
    print(f"Filled rows: {d['is_filled'].mean():.1%}")
    return d

if __name__ == "__main__":
    raw = pd.read_pickle("merged_raw.pkl")
    cleaned = clean(raw)
    print(cleaned.groupby("mobility").agg(rows=("CQI", "size"),
          traces=("trace_id", "nunique"), segments=("seg_uid", "nunique")))
    cleaned.to_pickle("cleaned.pkl")