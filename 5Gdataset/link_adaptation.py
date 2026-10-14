"""Backend: CQI -> modulation / code rate / spectral efficiency, and replay summaries.
Run as a script to write results_link_adaptation.csv."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# 3GPP 4-bit CQI table (up to 64QAM). Check it matches your dataset's CQI definition.
# cqi, modulation, code rate x 1024
_ROWS = [
    (1, "QPSK", 78), (2, "QPSK", 120), (3, "QPSK", 193), (4, "QPSK", 308),
    (5, "QPSK", 449), (6, "QPSK", 602), (7, "16QAM", 378), (8, "16QAM", 490),
    (9, "16QAM", 616), (10, "64QAM", 466), (11, "64QAM", 567), (12, "64QAM", 666),
    (13, "64QAM", 772), (14, "64QAM", 873), (15, "64QAM", 948),
]
_BITS = {"QPSK": 2, "16QAM": 4, "64QAM": 6}


def _build_table():
    t = pd.DataFrame(_ROWS, columns=["cqi", "modulation", "rate_x1024"])
    t["bits_per_symbol"] = t["modulation"].map(_BITS)
    t["code_rate"] = t["rate_x1024"] / 1024
    t["spectral_efficiency"] = t["bits_per_symbol"] * t["code_rate"]
    zero = pd.DataFrame([{"cqi": 0, "modulation": "None (out of range)", "rate_x1024": 0,
                          "bits_per_symbol": 0, "code_rate": 0.0, "spectral_efficiency": 0.0}])
    return pd.concat([zero, t], ignore_index=True).set_index("cqi")


TABLE = _build_table()
TAGS = ["reactive", "predictive", "actual"]
COLS = ["cqi", "modulation", "bits_per_symbol", "code_rate", "spectral_efficiency"]


def find_file(name):
    for d in (BASE, os.path.join(BASE, "outputs")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load_predictions():
    pred = pd.read_csv(find_file("test_predictions.csv"), parse_dates=["Timestamp"])
    pq = find_file("cleaned_5g.parquet")
    if pq:
        meta = pd.read_parquet(pq, columns=["trace_id", "content"]).drop_duplicates("trace_id")
        pred = pred.merge(meta, on="trace_id", how="left")
    else:
        pred["content"] = np.nan
    return pred


def detect_models(pred):
    return [c[5:] for c in pred.columns
            if c.startswith("pred_") and c != "pred_Persistence" and not c.endswith("_cqi")]


def dataset_path(app, mobility, content, trace_id):
    parts = [str(app), str(mobility)]
    if pd.notna(content) and content != "none":
        parts.append(str(content))
    return "/".join(parts + [f"{trace_id}.csv"])


def decide(cqi):
    """Round/clip CQI values and look up modulation, code rate and spectral efficiency."""
    c = np.clip(np.rint(np.asarray(cqi, dtype=float)), 0, int(TABLE.index.max())).astype(int)
    return TABLE.loc[c].reset_index()[COLS]


def outcome(chosen, actual):
    if chosen > actual:
        return "Over-estimate (risk of failed transmission)"
    if chosen < actual:
        return "Cautious (capacity left unused)"
    return "Matches the channel"


def replay(df, model):
    """df: rows of test_predictions.csv for ONE horizon, with a clean 0..n-1 index.
    Reactive uses the current CQI, predictive uses the model, actual is the real CQI at t+h."""
    sources = {"reactive": df["pred_Persistence"], "predictive": df[f"pred_{model}"],
               "actual": df["y_true"]}
    out = pd.DataFrame(index=df.index)
    for tag in TAGS:
        d = decide(sources[tag].to_numpy())
        for col in COLS:
            out[f"{col}_{tag}"] = d[col].to_numpy()
    for tag in ["reactive", "predictive"]:
        over = out[f"cqi_{tag}"] > out["cqi_actual"]
        out[f"over_{tag}"] = over
        # simplified failure model: a choice above the actual CQI fails and carries nothing
        out[f"effective_se_{tag}"] = np.where(over, 0.0, out[f"spectral_efficiency_{tag}"])
    return out


def summarize(rep, trace_ids=None):
    s = {"windows": len(rep), "ideal_se": rep["spectral_efficiency_actual"].mean()}
    for tag in ["reactive", "predictive"]:
        s[f"effective_se_{tag}"] = rep[f"effective_se_{tag}"].mean()
        s[f"pct_of_ideal_{tag}"] = (100 * s[f"effective_se_{tag}"] / s["ideal_se"]
                                    if s["ideal_se"] > 0 else np.nan)
        s[f"over_{tag}_pct"] = 100 * rep[f"over_{tag}"].mean()
    if trace_ids is not None:
        g = pd.DataFrame({"t": np.asarray(trace_ids),
                          "r": rep["effective_se_reactive"].to_numpy(),
                          "p": rep["effective_se_predictive"].to_numpy()}).groupby("t").mean()
        s["traces"] = len(g)
        s["traces_predictive_better"] = int((g["p"] > g["r"]).sum())
    return s


if __name__ == "__main__":
    pred = load_predictions()
    rows = []
    for h in sorted(pred["h"].unique()):
        ph = pred[pred["h"] == h].reset_index(drop=True)
        for model in detect_models(pred):
            rep = replay(ph, model)
            for mob in ["All"] + sorted(ph["mobility"].unique()):
                m = np.ones(len(ph), bool) if mob == "All" else (ph["mobility"] == mob).to_numpy()
                s = summarize(rep[m], ph.loc[m, "trace_id"])
                rows.append({"model": model, "h": int(h), "mobility": mob, **s})
    out = pd.DataFrame(rows)
    out.to_csv("results_link_adaptation.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(out.round(3).to_string(index=False))