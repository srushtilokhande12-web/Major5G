import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="5G CQI forecast evaluation", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [BASE, os.path.join(BASE, "outputs")]
EXPECTED = {
    "results_test.csv": "evaluate_test.py",
    "results_test_gains.csv": "evaluate_test.py",
    "test_predictions.csv": "evaluate_test.py",
}
MODEL_ORDER = ["Persistence", "XGBoost", "GRU"]
COLORS = {"Observed": "#222222", "Persistence": "#e08a1e",
          "XGBoost": "#1f77b4", "GRU": "#2ca02c"}
MOBS = ["All", "Driving", "Static"]
HORIZONS = [1, 3, 5]


def find(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def color_enc(title):
    return alt.Color("Series:N", title=title,
                     scale=alt.Scale(domain=list(COLORS),
                                     range=list(COLORS.values())))


# ---------- missing-file check ----------
missing = [(f, s) for f, s in EXPECTED.items() if find(f) is None]
if missing:
    st.title("5G CQI forecasting: test-set evaluation")
    st.error("Some result files were not found. Create them first, then reload this page.")
    st.table(pd.DataFrame(missing, columns=["Missing file", "Created by"]))
    st.markdown(
        "- `evaluate_test.py` needs the trained models from `train_xgb.py` and the "
        "split from `split_data.py`.\n"
        "- Put the CSV files next to `app.py` or in an `outputs/` folder."
    )
    st.stop()


@st.cache_data
def load():
    res = pd.read_csv(find("results_test.csv"))
    gains = pd.read_csv(find("results_test_gains.csv"))
    pred = pd.read_csv(find("test_predictions.csv"), parse_dates=["Timestamp"])
    return res, gains, pred


res, gains, pred = load()
models = [m for m in MODEL_ORDER if m in res["model"].unique()]
ml_models = [m for m in models if m != "Persistence"]


def traces_in(mob):
    if mob == "All":
        return pred["trace_id"].nunique()
    return pred.loc[pred["mobility"] == mob, "trace_id"].nunique()


# ---------- header ----------
st.title("5G CQI forecasting: test-set evaluation")
st.caption("Models predict CQI 1, 3 and 5 seconds ahead from the last few seconds of "
           "measurements. This page only reads saved results; nothing is retrained.")

st.info(
    "**How to read the results**\n"
    "- **MAE and RMSE:** lower is better (units are CQI levels, scale 0-15).\n"
    "- **Within ±1:** higher is better. Predictions are rounded and clipped to 0-15 "
    "before comparing.\n"
    "- **Persistence** predicts that CQI stays at its current value. A model is only "
    "useful if it beats it.\n"
    f"- **Limited test set:** {traces_in('All')} test traces in total "
    f"({traces_in('Driving')} Driving, {traces_in('Static')} Static). "
    "Windows from one trace overlap heavily, so group results rest on few traces."
)

tab1, tab2 = st.tabs(["Test metrics", "Trace explorer"])

# =====================================================================
with tab1:
    c1, c2 = st.columns(2)
    mob = c1.radio("Mobility group", MOBS, horizontal=True)
    h = c2.radio("Forecast horizon", HORIZONS, horizontal=True,
                 format_func=lambda x: f"{x} s")

    sub = res[(res["h"] == h) & (res["mobility"] == mob)].set_index("model").reindex(models)
    st.subheader(f"Horizon {h} s, {mob}")
    st.caption(f"{int(sub['n'].iloc[0]):,} test windows from {traces_in(mob)} traces.")

    # compact summary: each model vs Persistence
    base = sub.loc["Persistence"]
    for m in ml_models:
        st.markdown(f"**{m} vs Persistence**")
        cols = st.columns(3)
        for col, key, label in [(cols[0], "MAE", "MAE"), (cols[1], "RMSE", "RMSE")]:
            d = sub.loc[m, key] - base[key]
            col.metric(label, f"{sub.loc[m, key]:.3f}", f"{d:+.3f}",
                       delta_color="off" if abs(d) < 0.0005 else "inverse")
        d = (sub.loc[m, "within1"] - base["within1"]) * 100
        cols[2].metric("Within ±1", f"{sub.loc[m, 'within1'] * 100:.1f}%", f"{d:+.1f} pts",
                       delta_color="off" if abs(d) < 0.05 else "normal")

    table = sub[["MAE", "RMSE", "within1"]].copy()
    table["within1"] = table["within1"] * 100
    table.columns = ["MAE (lower is better)", "RMSE (lower is better)",
                     "Within ±1 % (higher is better)"]
    st.dataframe(table.round(3), use_container_width=True)

    # ---- gain over persistence ----
    st.subheader("MAE improvement over Persistence")
    g = gains[(gains["h"] == h) & (gains["mobility"] == mob)].copy()
    if g.empty:
        st.write("No gain rows for this selection.")
    else:
        g["95% CI"] = g.apply(lambda r: f"[{r.CI_low:.3f}, {r.CI_high:.3f}]", axis=1)
        g["CI excludes 0"] = np.where((g["CI_low"] > 0) | (g["CI_high"] < 0), "Yes", "No")
        out = g[["model", "MAE_gain", "95% CI", "traces_better", "CI excludes 0"]]
        out.columns = ["Model", "MAE gain", "95% CI", "Traces better than Persistence",
                       "CI excludes 0"]
        st.dataframe(out.round(3), hide_index=True, use_container_width=True)
    st.caption("Gain = Persistence MAE minus model MAE (positive means the model is "
               "better). The 95% interval comes from resampling whole traces. If it "
               "includes 0, the gain is not confirmed. At 1 s the model is effectively "
               "Persistence, so no gain is expected.")

    gh = gains[gains["h"] == h].copy()
    gh["Series"] = gh["model"]
    gh["Group"] = gh["mobility"] + " - " + gh["model"]
    order = [f"{mb} - {m}" for mb in MOBS for m in ml_models]
    ybase = alt.Chart(gh).encode(y=alt.Y("Group:N", sort=order, title=None))
    rule = ybase.mark_rule(strokeWidth=3).encode(
        x=alt.X("CI_low:Q", title="MAE gain over Persistence (CQI levels)"),
        x2="CI_high:Q", color=color_enc("Model"))
    pts = ybase.mark_point(filled=True, size=90).encode(
        x="MAE_gain:Q", color=color_enc("Model"),
        tooltip=["Group", "MAE_gain", "CI_low", "CI_high", "traces_better"])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="x:Q")
    st.altair_chart(alt.layer(rule, pts, zero).properties(
        height=70 * max(1, len(order) // 2) + 60,
        title=f"Gain with 95% interval, horizon {h} s"), use_container_width=True)

    # ---- metric across horizons ----
    st.subheader(f"All horizons, {mob}")
    metric = st.selectbox("Metric", ["MAE", "RMSE", "Within ±1 (%)"])
    d = res[res["mobility"] == mob].copy()
    d["Series"] = d["model"]
    d["value"] = d["within1"] * 100 if metric.startswith("Within") else d[metric]
    chart = alt.Chart(d).mark_bar().encode(
        x=alt.X("Series:N", title=None, axis=alt.Axis(labels=False)),
        y=alt.Y("value:Q", title=metric),
        color=color_enc("Model"),
        column=alt.Column("h:O", title="Forecast horizon (s)"),
        tooltip=["Series", "h", "value"]).properties(width=140, height=260)
    st.altair_chart(chart)

# =====================================================================
with tab2:
    c1, c2 = st.columns([2, 1])
    hh = c2.radio("Forecast horizon", HORIZONS, horizontal=True,
                  format_func=lambda x: f"{x} s", key="h_explorer")
    ph = pred[pred["h"] == hh]
    info = ph.groupby("trace_id").agg(mobility=("mobility", "first"),
                                      app=("app", "first"), n=("y_true", "size"))
    labels = {t: f"{t}  ({r.mobility}, {r.app}, {r.n} windows)"
              for t, r in info.iterrows()}
    trace = c1.selectbox("Test trace", list(labels), format_func=lambda t: labels[t])

    d = ph[ph["trace_id"] == trace].sort_values("Timestamp").copy()
    d["target_time"] = d["Timestamp"] + pd.to_timedelta(hh, unit="s")

    series_cols = {"Observed": "y_true"}
    series_cols.update({m: f"pred_{m}" for m in models})
    shown = st.multiselect("Series to show", list(series_cols),
                           default=list(series_cols))

    long = d.melt(id_vars=["seg_uid", "target_time"],
                  value_vars=[series_cols[s] for s in shown],
                  var_name="col", value_name="CQI")
    inv = {v: k for k, v in series_cols.items()}
    long["Series"] = long["col"].map(inv)

    line = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=alt.X("target_time:T", title="Time the CQI refers to"),
        y=alt.Y("CQI:Q", scale=alt.Scale(domain=[0, 15]), title="CQI"),
        color=color_enc("Series"),
        detail="seg_uid:N",
        tooltip=["Series", "target_time", "CQI"]).properties(height=380).interactive()
    st.altair_chart(line, use_container_width=True)
    st.caption(f"Each prediction is made {hh} s earlier and plotted at the time it refers "
               "to, so it lines up with the observed CQI. Persistence therefore looks "
               "like a delayed copy of the observed line. Lines break where the trace has "
               "gaps (non-5G stretches). Drag to pan, scroll to zoom.")

    # per-trace metrics
    rows = []
    for m in models:
        p = d[f"pred_{m}"].to_numpy()
        y = d["y_true"].to_numpy()
        pr = np.clip(np.rint(p), 0, 15)
        rows.append({"Model": m,
                     "MAE": np.abs(p - y).mean(),
                     "RMSE": np.sqrt(((p - y) ** 2).mean()),
                     "Within ±1 %": (np.abs(pr - y) <= 1).mean() * 100})
    st.markdown(f"**This trace ({info.loc[trace, 'mobility']}, {len(d):,} windows)**")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption("One trace is a small sample. Use the Test metrics tab for overall results.")