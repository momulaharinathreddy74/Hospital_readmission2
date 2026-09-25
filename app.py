"""Streamlit dashboard for the readmission & cost analytics pipeline.

Run the pipeline first (python3 run_all.py), then:
    streamlit run app.py
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from src import config
from src.assumptions import ASSUMPTIONS, DATASET_YEARS, value
from src.savings import department_table
from src.tiering import capture_at

TARGET = "readmit_30"
TEAL, RED, GREEN, GRAY = "#1F6F78", "#B8333A", "#2E7D5B", "#B9C4C6"

st.set_page_config(page_title="Readmission risk and cost", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    need = ["encounters_clean.csv", "predictions_calibrated.csv", "test_tiered.csv"]
    missing = [f for f in need if not (config.DATA_PROCESSED / f).exists()]
    if missing:
        return None
    clean = pd.read_csv(config.DATA_PROCESSED / "encounters_clean.csv")
    preds = pd.read_csv(config.DATA_PROCESSED / "predictions_calibrated.csv")
    tiered = pd.read_csv(config.DATA_PROCESSED / "test_tiered.csv")
    return clean, preds, tiered


@st.cache_data
def capture_curve(y, score):
    """Share of readmissions captured when enrolling the top p% (p = 1..100)."""
    pcts = np.arange(1, 101)
    return pd.DataFrame({"pct": pcts, "capture": [capture_at(y, score, p) for p in pcts]})


def money(v, per_year=False):
    v = v / DATASET_YEARS if per_year else v
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v / 1e6:.1f}M" if v >= 1e6 else f"{sign}${v / 1e3:,.0f}K"


data = load()
if data is None:
    st.error("Processed data not found. From the project folder, run "
             "`python3 run_all.py`, then reload this page.")
    st.stop()
clean, preds, tiered = data
test = preds[preds["split"] == "test"]
n_enc, n_readm = len(clean), int(clean[TARGET].sum())
curve = capture_curve(tiered[TARGET].to_numpy(), tiered["raw_score"].to_numpy())

st.title("Readmission risk and cost")
st.caption(f"Diabetes 130-US Hospitals, 1999-2008: {n_enc:,} encounters, "
           f"{n_readm:,} readmitted within 30 days ({n_readm / n_enc:.1%}).")

sim, tiers_tab, dept_tab, model_tab, method_tab = st.tabs(
    ["Savings simulator", "Risk tiers", "Departments", "Model quality", "Method"])

# ================================================================ simulator
with sim:
    st.subheader("Would a transitional-care program pay for itself?")
    st.write("Enroll the highest-risk patients in a follow-up program. "
             "Adjust the assumptions to see when the savings outweigh the cost.")

    c1, c2, c3, c4 = st.columns(4)
    pct = c1.slider("Patients enrolled (highest risk first)", 1, 100, 20, format="%d%%")
    eff = c2.slider("Readmissions the program prevents", 5, 30, 15, format="%d%%",
                    help=ASSUMPTIONS["effectiveness_low"]["source"]) / 100
    icost = c3.slider("Program cost per patient", 100, 1500,
                      value("intervention_cost_per_patient"), step=50, format="$%d",
                      help=ASSUMPTIONS["intervention_cost_per_patient"]["source"])
    cpr = c4.slider("Cost of one readmission", 10_000, 20_000,
                    value("cost_per_readmission"), step=100, format="$%d",
                    help=ASSUMPTIONS["cost_per_readmission"]["source"])
    per_year = st.toggle(f"Show per year (dataset covers {DATASET_YEARS} years)", value=False)

    curve["enrolled"] = n_enc * curve["pct"] / 100
    curve["prevented"] = n_readm * curve["capture"] * eff
    curve["gross"] = curve["prevented"] * cpr
    curve["program"] = curve["enrolled"] * icost
    curve["net"] = curve["gross"] - curve["program"]
    row = curve.loc[curve["pct"] == pct].iloc[0]
    best = curve.loc[curve["net"].idxmax()]
    breakeven = row["gross"] / row["enrolled"]

    net_color = GREEN if row["net"] >= 0 else RED
    verdict = "Pays for itself" if row["net"] >= 0 else "Costs more than it saves"
    st.markdown(
        f"<div style='padding:1.2rem 0 0.4rem'>"
        f"<div style='font-size:3.2rem;font-weight:700;line-height:1;color:{net_color}'>"
        f"{money(row['net'], per_year)}</div>"
        f"<div style='font-size:1.1rem;margin-top:.4rem'>{verdict}: net savings "
        f"{'per year' if per_year else 'over the dataset'}</div></div>",
        unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Readmissions prevented", f"{row['prevented'] / (DATASET_YEARS if per_year else 1):,.0f}")
    k2.metric("Gross avoidable cost", money(row["gross"], per_year))
    k3.metric("Program cost", money(row["program"], per_year))
    k4.metric("Break-even cost per patient", f"${breakeven:,.0f}")

    plot = curve.assign(net_m=curve["net"] / 1e6 / (DATASET_YEARS if per_year else 1))
    line = alt.Chart(plot).mark_line(color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("pct:Q", title="% of patients enrolled"),
        y=alt.Y("net_m:Q", title="Net savings ($M)"),
        tooltip=[alt.Tooltip("pct:Q", title="% enrolled"),
                 alt.Tooltip("net_m:Q", title="Net ($M)", format=".2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#1C2B2D").encode(y="y:Q")
    here = alt.Chart(plot[plot["pct"] == pct]).mark_point(
        size=120, filled=True, color=net_color).encode(x="pct:Q", y="net_m:Q")
    st.altair_chart((line + zero + here).properties(height=300), width="stretch")
    st.write(f"With these assumptions, net savings peak when the top **{int(best['pct'])}%** "
             f"are enrolled ({money(best['net'], per_year)}). Beyond that, each extra patient "
             "costs more than the readmissions they would avoid.")

# ================================================================ tiers
with tiers_tab:
    st.subheader("Who is at highest risk")
    t = (tiered.groupby("tier", observed=True)
         .agg(Patients=(TARGET, "size"), Readmissions=(TARGET, "sum"),
              Predicted=("cal_score", "mean"), Actual=(TARGET, "mean")))
    t = t.reindex([n for n, _ in config.RISK_TIERS])
    t["Share of patients"] = t["Patients"] / t["Patients"].sum()
    t["Share of readmissions"] = t["Readmissions"] / t["Readmissions"].sum()
    t["Lift"] = t["Actual"] / tiered[TARGET].mean()
    st.dataframe(t.style.format({"Predicted": "{:.1%}", "Actual": "{:.1%}",
                                 "Share of patients": "{:.1%}",
                                 "Share of readmissions": "{:.1%}", "Lift": "{:.2f}x",
                                 "Patients": "{:,}", "Readmissions": "{:,}"}),
                 width="stretch")
    top20 = curve.loc[curve["pct"] == 20, "capture"].iloc[0]
    st.write(f"The top 20% of patients account for **{top20:.1%}** of readmissions, "
             f"about {top20 / 0.2:.1f}x what random selection would find. "
             "Held-out test set.")
    g = pd.concat([pd.DataFrame({"pct": [0], "capture": [0.0]}), curve[["pct", "capture"]]])
    gains = alt.Chart(g).mark_line(color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("pct:Q", title="% of patients enrolled"),
        y=alt.Y("capture:Q", title="% of readmissions reached", axis=alt.Axis(format="%")))
    rand = alt.Chart(pd.DataFrame({"pct": [0, 100], "capture": [0, 1]})).mark_line(
        color=GRAY, strokeDash=[5, 4]).encode(x="pct:Q", y="capture:Q")
    st.altair_chart((gains + rand).properties(height=320), width="stretch")
    st.caption("Dashed line: selecting patients at random.")

# ================================================================ departments
with dept_tab:
    st.subheader("Departments, adjusted for patient mix")
    st.write("Observed ÷ expected readmissions. Above 1 means more readmissions than "
             "the department's patients would predict. Colored only when the 95% "
             "interval excludes 1.")
    scored = preds[preds["split"].isin(["val", "test"])]
    dept = department_table(clean, scored, cpr).reset_index()
    d = dept.dropna(subset=["oe_ratio"])
    d["color"] = d["flag"].map({"Worse than expected": RED,
                                "Better than expected": GREEN}).fillna(GRAY)
    order = d.sort_values("oe_ratio", ascending=False)["specialty"].tolist()
    bars = alt.Chart(d).mark_bar().encode(
        y=alt.Y("specialty:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("oe_ratio:Q", title="Observed / expected"),
        color=alt.Color("color:N", scale=None),
        tooltip=["specialty", alt.Tooltip("oe_ratio:Q", format=".2f"),
                 alt.Tooltip("oe_low:Q", format=".2f", title="CI low"),
                 alt.Tooltip("oe_high:Q", format=".2f", title="CI high")])
    ci = alt.Chart(d).mark_rule(color="#5A6B6D").encode(
        y=alt.Y("specialty:N", sort=order), x="oe_low:Q", x2="oe_high:Q")
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart((bars + ci + one).properties(height=420), width="stretch")

    show = dept[["specialty", "encounters", "readmit_rate", "avg_los", "bed_days",
                 "readmission_cost", "readmit_cost_per_encounter", "oe_ratio", "flag"]]
    show.columns = ["Department", "Encounters", "Readmit rate", "Avg stay (days)",
                    "Bed-days", "Readmission cost", "Per encounter", "O/E", "Flag"]
    st.dataframe(show.style.format({"Encounters": "{:,}", "Readmit rate": "{:.1%}",
                                    "Avg stay (days)": "{:.1f}", "Bed-days": "{:,}",
                                    "Readmission cost": "${:,.0f}",
                                    "Per encounter": "${:,.0f}", "O/E": "{:.2f}"},
                                   na_rep="n/a"),
                 hide_index=True, width="stretch")
    st.caption(f"Readmission cost uses ${cpr:,} per readmission (set in the simulator). "
               "49% of encounters have no recorded department.")

# ================================================================ model
with model_tab:
    st.subheader("How well the model works")
    auc = roc_auc_score(test[TARGET], test["raw_score"])
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC (held-out test)", f"{auc:.3f}", help="Published range for this dataset: about 0.64-0.70.")
    m2.metric("Average predicted risk", f"{test['cal_score'].mean():.1%}",
              delta=f"was {test['raw_score'].mean():.1%} before calibration", delta_color="off")
    m3.metric("Actual readmission rate", f"{test[TARGET].mean():.1%}")

    bins = pd.qcut(test["raw_score"].rank(method="first"), 10, labels=False)
    rel = pd.concat([
        test.assign(b=bins).groupby("b").agg(pred=("raw_score", "mean"), actual=(TARGET, "mean")).assign(Scores="Before calibration"),
        test.assign(b=bins).groupby("b").agg(pred=("cal_score", "mean"), actual=(TARGET, "mean")).assign(Scores="After isotonic calibration")])
    pts = alt.Chart(rel).mark_line(point=True).encode(
        x=alt.X("pred:Q", title="Predicted risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
        y=alt.Y("actual:Q", title="Actual readmission rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("Scores:N", scale=alt.Scale(range=[GRAY, TEAL]), legend=alt.Legend(orient="top")))
    diag = alt.Chart(pd.DataFrame({"x": [0, 0.8]})).mark_line(color="#1C2B2D", strokeDash=[4, 3]).encode(x="x:Q", y="x:Q")
    st.altair_chart((diag + pts).properties(height=380), width="stretch")
    st.caption("Points on the dashed line mean predicted risk matches reality. "
               "Deciles of the held-out test set.")

# ================================================================ method
with method_tab:
    st.subheader("How this was built")
    st.markdown(f"""
1. **Clean and map codes.** Admission, discharge and source IDs grouped using UB-04
   code meanings; ICD-9 diagnoses grouped into 9 clinical categories. Patients who
   died or went to hospice were excluded.
2. **Check for leakage.** A "total encounters per patient" feature correlated 0.235
   with readmission because it counts future visits. It was caught by a
   point-in-time test and removed; it would have inflated AUC by about 0.11.
3. **Train by patient.** 60/20/20 split with no patient in more than one set.
   Gradient boosting, tuned on validation, scored once on test.
4. **Calibrate.** Class weighting had inflated predicted risk to about 45% against
   an 11% actual rate. Isotonic regression fitted on validation fixed it without
   changing the ranking.
5. **Tier and simulate.** Patients ranked by risk; savings simulated from sourced
   assumptions, with program cost included.

**Assumptions and sources**
""")
    for k, a in ASSUMPTIONS.items():
        v = f"{a['value']:.0%}" if a["unit"] == "%" else f"${a['value']:,}"
        st.markdown(f"- **{k.replace('_', ' ').capitalize()}:** {v}. {a['source']}")
    st.markdown("""
**Limitations.** Data is from 1999-2008 and covers diabetic encounters only. Costs
are national averages, not hospital-specific. The simulation assumes the program
works equally well at every risk level, which a real pilot would need to test.
Full reports are in the `reports/` folder.
""")
