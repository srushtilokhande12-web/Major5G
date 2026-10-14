import json
import altair as alt
import pandas as pd
import streamlit as st
import link_adaptation as la

st.set_page_config(page_title="Reactive vs predictive 5G link adaptation", layout="wide")

REACTIVE, ACTUAL = "#f0a030", "#9a9a9a"
MODEL_COLORS = {"XGBoost": "#4da3ff", "GRU": "#59d98e"}

st.title("Reactive vs predictive 5G link adaptation")

if la.find_file("test_predictions.csv") is None:
    st.error("`test_predictions.csv` was not found. It is created by `evaluate_test.py` "
             "(run `train_xgb.py` first). Put it next to `app.py` or in `outputs/`.")
    st.stop()


@st.cache_data
def load():
    return la.load_predictions()


@st.cache_data
def overall(h, model):
    p = load()
    ph = p[p["h"] == h].reset_index(drop=True)
    return la.summarize(la.replay(ph, model), ph["trace_id"])


pred = load()
models = la.detect_models(pred)
if not models:
    st.error("No model prediction columns (pred_<Model>) found in test_predictions.csv.")
    st.stop()
horizons = sorted(int(x) for x in pred["h"].unique())

cfg = la.find_file("best_config.json")
W = json.load(open(cfg)).get("W") if cfg else None
st.caption("Offline replay of recorded test traces, which the model never saw in training. "
           + (f"The model reads the last {W} s of measurements. " if W else "")
           + "Nothing is retrained here.")

# ---------- controls ----------
info = pred.drop_duplicates("trace_id").set_index("trace_id")
labels = {t: f"{la.dataset_path(r['app'], r['mobility'], r['content'], t)}   [{r['mobility']}]"
          for t, r in info.iterrows()}
c1, c2, c3 = st.columns([4, 1.6, 1.6])
trace = c1.selectbox("Test trace (dataset file)", sorted(labels), format_func=lambda t: labels[t])
h = c2.radio("Prediction horizon", horizons, horizontal=True, format_func=lambda x: f"{x} s")
model = c3.radio("Model", models, horizontal=True)
PRED_COLOR = MODEL_COLORS.get(model, "#4da3ff")

d = pred[(pred["h"] == h) & (pred["trace_id"] == trace)].sort_values("Timestamp").reset_index(drop=True)
rep = la.replay(d, model)
n = len(d)
i = st.slider("Moment in the trace (window number)", 0, n - 1, min(n - 1, 60), key=f"i_{trace}_{h}")
t0 = d.loc[i, "Timestamp"]
st.caption(f"{n:,} windows in this trace. Selected moment: {t0}. Each scheme picks a CQI for the "
           f"transmission {h} s later.")

# ---------- comparison table ----------
r = rep.iloc[i]


def column(tag):
    vals = [str(int(r[f"cqi_{tag}"])), r[f"modulation_{tag}"],
            str(int(r[f"bits_per_symbol_{tag}"])), f"{r[f'code_rate_{tag}']:.3f}",
            f"{r[f'spectral_efficiency_{tag}']:.2f}"]
    if tag == "actual":
        vals.append("Reference: best supportable choice")
    else:
        vals.append(la.outcome(r[f"cqi_{tag}"], r["cqi_actual"]))
    return vals


table = pd.DataFrame(
    {"Reactive (current CQI)": column("reactive"),
     f"Predictive ({model}, +{h} s)": column("predictive"),
     "Actual (CQI at +%d s)" % h: column("actual")},
    index=["CQI", "Modulation used", "Bits per symbol", "Code rate",
           "Spectral efficiency (bits/s/Hz)", "Outcome"])
st.table(table)

# ---------- charts ----------
seg = (d["seg_uid"] == d.loc[i, "seg_uid"]).to_numpy()
times = (d.loc[seg, "Timestamp"] + pd.to_timedelta(h, unit="s")).to_numpy()
names = {"reactive": "Reactive", "predictive": f"Predictive ({model})",
         "actual": "Actual (best supportable)"}
scale = alt.Scale(domain=list(names.values()), range=[REACTIVE, PRED_COLOR, ACTUAL])
dash = alt.Scale(domain=list(names.values()), range=[[1, 0], [1, 0], [4, 3]])
sel = pd.DataFrame({"time": [t0 + pd.Timedelta(seconds=h)]})


def chart(prefix, title, axis=None):
    long = pd.concat([pd.DataFrame({"time": times, "value": rep.loc[seg, f"{prefix}_{tag}"].to_numpy(),
                                    "Series": names[tag]}) for tag in la.TAGS])
    line = alt.Chart(long).mark_line(interpolate="step-after", strokeWidth=2.2).encode(
        x=alt.X("time:T", title="Time the decision applies to"),
        y=alt.Y("value:Q", title=title, axis=axis),
        color=alt.Color("Series:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("Series:N", scale=dash, legend=None),
        tooltip=["Series", "time", "value"])
    rule = alt.Chart(sel).mark_rule(strokeDash=[5, 4], color="white").encode(x="time:T")
    return alt.layer(line, rule).properties(height=280, title=title)


bits = sorted(int(b) for b in la.TABLE["bits_per_symbol"].unique())
g1, g2 = st.columns(2)
g1.altair_chart(chart("bits_per_symbol", "Modulation order (bits per symbol)",
                      alt.Axis(values=bits)), use_container_width=True)
g2.altair_chart(chart("spectral_efficiency", "Spectral efficiency (bits/s/Hz)"),
                use_container_width=True)
st.caption("Reactive reuses the CQI measured at the decision time, so it lags the channel. "
           "Where the reactive or predictive line is above the Actual line, the choice was "
           "too aggressive. The white dashed line marks the selected moment.")


# ---------- summary ----------
def show(s, title):
    st.subheader(title)
    k = st.columns(5)
    k[0].metric("Ideal spectral efficiency", f"{s['ideal_se']:.2f}",
                help="Average efficiency if every choice matched the actual CQI.")
    k[1].metric("Reactive (effective)", f"{s['effective_se_reactive']:.2f}",
                help=f"{s['pct_of_ideal_reactive']:.1f}% of ideal")
    k[2].metric("Predictive (effective)", f"{s['effective_se_predictive']:.2f}",
                f"{s['effective_se_predictive'] - s['effective_se_reactive']:+.3f} vs reactive",
                help=f"{s['pct_of_ideal_predictive']:.1f}% of ideal")
    k[3].metric("Over-estimates, reactive", f"{s['over_reactive_pct']:.1f}%")
    k[4].metric("Over-estimates, predictive", f"{s['over_predictive_pct']:.1f}%",
                f"{s['over_predictive_pct'] - s['over_reactive_pct']:+.1f} pts", delta_color="inverse")
    if "traces" in s:
        st.caption(f"Predictive had higher effective efficiency on "
                   f"{s['traces_predictive_better']} of {s['traces']} traces.")


show(la.summarize(rep, d["trace_id"]), f"This trace, horizon {h} s")
show(overall(h, model), f"All {pred['trace_id'].nunique()} test traces, horizon {h} s")

st.info("**Effective spectral efficiency** = the chosen efficiency if the choice was not above "
        "the actual CQI, otherwise 0 (a failed transmission). Higher is better; over-estimates "
        "should be lower. This is a simplified replay, not a throughput simulation, and the test "
        "set has few traces, so group differences are uncertain.")

with st.expander("CQI to modulation table used (from link_adaptation.py)"):
    st.dataframe(la.TABLE.reset_index(), hide_index=True, use_container_width=True)