"""Step 7: department-level cost/efficiency analysis and savings simulation.

Part A - Departments (medical specialty):
  volume, bed-days, length of stay, readmission rate, readmission cost, and a
  risk-adjusted observed/expected (O/E) ratio, the same idea CMS uses in the
  Hospital Readmissions Reduction Program.

Part B - Savings simulation:
  If the top risk tiers receive a transitional-care intervention, how much
  readmission cost is avoided (gross), what does the program cost, and what
  is left (net)? Plus sensitivity analysis and the best targeting depth.

Report: reports/07_savings_report.md (+ figures)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src import config
from src.assumptions import ASSUMPTIONS, DATASET_YEARS, value
from src.tiering import capture_at

TARGET = "readmit_30"
TARGET_TIERS = ["Very high", "High"]
MIN_SCORED = 250               # departments with fewer scored encounters get no O/E


# ================================================================ Part A
def department_table(clean, scored, cost_per_readmission):
    """clean: all encounters. scored: out-of-sample encounters with cal_score."""
    d = clean.groupby("specialty").agg(
        encounters=(TARGET, "size"), readmissions=(TARGET, "sum"),
        readmit_rate=(TARGET, "mean"), avg_los=("time_in_hospital", "mean"),
        bed_days=("time_in_hospital", "sum"))
    d["readmission_cost"] = d["readmissions"] * cost_per_readmission
    d["readmit_cost_per_encounter"] = d["readmission_cost"] / d["encounters"]

    # Risk adjustment: compare observed readmissions with what the calibrated
    # model expects given each department's patient mix.
    oe = scored.groupby("specialty").agg(n_scored=(TARGET, "size"),
                                         observed=(TARGET, "sum"),
                                         expected=("cal_score", "sum"))
    oe["oe_ratio"] = oe["observed"] / oe["expected"]
    # Exact Poisson 95% CI on the observed count, divided by expected
    o = oe["observed"]
    oe["oe_low"] = chi2.ppf(0.025, 2 * o) / 2 / oe["expected"]
    oe["oe_high"] = chi2.ppf(0.975, 2 * (o + 1)) / 2 / oe["expected"]
    oe["flag"] = np.select([oe["oe_low"] > 1, oe["oe_high"] < 1],
                           ["Worse than expected", "Better than expected"], "")
    d = d.join(oe[["n_scored", "oe_ratio", "oe_low", "oe_high", "flag"]])
    small = d["n_scored"] < MIN_SCORED
    d.loc[small, ["oe_ratio", "oe_low", "oe_high"]] = np.nan
    d.loc[small, "flag"] = ""
    return d.sort_values("readmission_cost", ascending=False)


# ================================================================ Part B
def simulate(n_encounters, n_readmissions, share_targeted, capture_rate,
             effectiveness, cost_per_readmission, intervention_cost):
    enrolled = n_encounters * share_targeted
    reached = n_readmissions * capture_rate
    prevented = reached * effectiveness
    gross = prevented * cost_per_readmission
    program = enrolled * intervention_cost
    return {"enrolled": enrolled, "readmissions_reached": reached,
            "prevented": prevented, "gross_savings": gross,
            "program_cost": program, "net_savings": gross - program,
            "breakeven_cost_per_patient": gross / enrolled}


def net_by_depth(y, score, n_enc, n_readm, eff, cpr, icost, pcts=range(1, 101)):
    rows = []
    for p in pcts:
        cap = capture_at(y, score, p)
        r = simulate(n_enc, n_readm, p / 100, cap, eff, cpr, icost)
        rows.append({"pct_targeted": p, "capture": cap, **r})
    return pd.DataFrame(rows)


def sensitivity_grid(n_enc, n_readm, share, cap):
    effs = [0.10, 0.15, 0.20, 0.27]
    icosts = [250, 500, 750, 1000]
    grid = pd.DataFrame(index=[f"{e:.0%}" for e in effs],
                        columns=[f"${c:,}" for c in icosts], dtype=float)
    for e in effs:
        for c in icosts:
            grid.loc[f"{e:.0%}", f"${c:,}"] = simulate(
                n_enc, n_readm, share, cap, e, value("cost_per_readmission"), c
            )["net_savings"]
    return grid


# ================================================================ plots
def plot_departments(dept, path):
    d = dept.dropna(subset=["oe_ratio"]).sort_values("oe_ratio")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = d["flag"].map({"Worse than expected": "crimson",
                            "Better than expected": "seagreen"}).fillna("lightgray")
    err = [d["oe_ratio"] - d["oe_low"], d["oe_high"] - d["oe_ratio"]]
    ax.barh(d.index, d["oe_ratio"], color=colors, xerr=err, capsize=3,
            error_kw={"lw": 1, "ecolor": "dimgray"})
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set(xlabel="Observed / expected readmissions (risk-adjusted)",
           title="Department readmission performance (95% CI)\n"
                 "(colored = CI excludes 1; gray = not significant)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_net_curve(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, c in curves.items():
        ax.plot(c["pct_targeted"], c["net_savings"] / 1e6, label=label)
        best = c.loc[c["net_savings"].idxmax()]
        ax.scatter(best["pct_targeted"], best["net_savings"] / 1e6, zorder=5)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(20, color="gray", ls=":", lw=1)
    ax.set(xlabel="% of patients enrolled (highest risk first)",
           ylabel="Net savings ($M, whole dataset)",
           title="Net savings vs. targeting depth\n($500/patient program cost)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ================================================================ main
def money(v):
    return f"-${abs(v) / 1e6:.1f}M" if v < 0 else f"${v / 1e6:.1f}M"


def main():
    clean = pd.read_csv(config.DATA_PROCESSED / "encounters_clean.csv")
    preds = pd.read_csv(config.DATA_PROCESSED / "predictions_calibrated.csv")
    tiered = pd.read_csv(config.DATA_PROCESSED / "test_tiered.csv")
    cpr, icost = value("cost_per_readmission"), value("intervention_cost_per_patient")
    eff_lo, eff_hi = value("effectiveness_low"), value("effectiveness_high")

    # ---- Part A: out-of-sample scores only (validation + test)
    scored = preds[preds["split"].isin(["val", "test"])]
    dept = department_table(clean, scored, cpr)

    # ---- Part B: rates measured on test, applied to the full population
    n_enc, n_readm = len(clean), int(clean[TARGET].sum())
    targeted = tiered["tier"].isin(TARGET_TIERS)
    share = targeted.mean()
    cap = tiered.loc[targeted, TARGET].sum() / tiered[TARGET].sum()
    lo = simulate(n_enc, n_readm, share, cap, eff_lo, cpr, icost)
    hi = simulate(n_enc, n_readm, share, cap, eff_hi, cpr, icost)
    grid = sensitivity_grid(n_enc, n_readm, share, cap)

    y, s = tiered[TARGET].to_numpy(), tiered["raw_score"].to_numpy()
    curves = {f"{e:.0%} effective": net_by_depth(y, s, n_enc, n_readm, e, cpr, icost)
              for e in (eff_lo, eff_hi)}

    figs = config.REPORTS / "figures"
    plot_departments(dept, figs / "department_oe.png")
    plot_net_curve(curves, figs / "net_savings_curve.png")

    pct = lambda v: f"{v:.1%}"
    L = ["# Step 7 - Department Cost Analysis & Savings Simulation", "",
         "## Assumptions", "| Input | Value | Source |", "|---|---|---|"]
    for k, a in ASSUMPTIONS.items():
        v = f"{a['value']:.0%}" if a["unit"] == "%" else f"${a['value']:,}"
        L.append(f"| {k} | {v} | {a['source']} |")
    L += ["", "## A. Department-level analysis",
          f"All {n_enc:,} encounters; O/E uses out-of-sample (validation + test) "
          "predictions only. O/E > 1 means more readmissions than the patient mix "
          "predicts.", "",
          "| Department | Encounters | Readmit rate | Avg LOS | Bed-days | "
          "Readmission cost | Per encounter | O/E (95% CI) | Flag |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in dept.iterrows():
        oe = ("n/a" if pd.isna(r["oe_ratio"]) else
              f"{r['oe_ratio']:.2f} ({r['oe_low']:.2f}-{r['oe_high']:.2f})")
        L.append(f"| {name} | {int(r['encounters']):,} | {pct(r['readmit_rate'])} | "
                 f"{r['avg_los']:.1f} | {int(r['bed_days']):,} | "
                 f"{money(r['readmission_cost'])} | ${r['readmit_cost_per_encounter']:,.0f} | "
                 f"{oe} | {r['flag']} |")
    L += ["", "![Department O/E](figures/department_oe.png)", "",
          "Notes: 49% of encounters have no recorded specialty (\"Unknown\"). "
          "Raw readmission rates partly reflect patient mix; the O/E column "
          "adjusts for it. A department is flagged only when its whole 95% CI "
          "is above or below 1.", "",
          "## B. Savings simulation",
          f"Population: {n_enc:,} encounters, {n_readm:,} readmissions. "
          f"Targeting the Very high + High tiers enrolls {pct(share)} of patients "
          f"and reaches {pct(cap)} of readmissions (measured on the test set).", "",
          "| | Low (10% effective) | High (20% effective) |", "|---|---|---|",
          f"| Patients enrolled | {lo['enrolled']:,.0f} | {hi['enrolled']:,.0f} |",
          f"| Readmissions reached | {lo['readmissions_reached']:,.0f} | {hi['readmissions_reached']:,.0f} |",
          f"| Readmissions prevented | {lo['prevented']:,.0f} | {hi['prevented']:,.0f} |",
          f"| **Gross avoidable cost** | **{money(lo['gross_savings'])}** | **{money(hi['gross_savings'])}** |",
          f"| Program cost (${icost}/patient) | {money(lo['program_cost'])} | {money(hi['program_cost'])} |",
          f"| **Net savings** | **{money(lo['net_savings'])}** | **{money(hi['net_savings'])}** |",
          f"| Break-even program cost per patient | ${lo['breakeven_cost_per_patient']:,.0f} | "
          f"${hi['breakeven_cost_per_patient']:,.0f} |", "",
          f"Time frame: the dataset spans {DATASET_YEARS} years across 130 hospitals, "
          f"so these are totals over the whole dataset, roughly "
          f"{money(lo['gross_savings'] / DATASET_YEARS)}-"
          f"{money(hi['gross_savings'] / DATASET_YEARS)} gross per year across all "
          "130 hospitals.", "",
          "## Sensitivity: net savings (top tiers enrolled)",
          "Rows = intervention effectiveness, columns = program cost per patient.", "",
          "| Effectiveness | " + " | ".join(grid.columns) + " |",
          "|---|" + "---|" * len(grid.columns)]
    for e, r in grid.iterrows():
        L.append(f"| {e} | " + " | ".join(money(v) for v in r) + " |")
    L += ["", "## Best targeting depth ($500/patient)", "| Scenario | Best % enrolled | Net savings |",
          "|---|---|---|"]
    for label, c in curves.items():
        b = c.loc[c["net_savings"].idxmax()]
        L.append(f"| {label} | top {int(b['pct_targeted'])}% | {money(b['net_savings'])} |")
    L += ["", "![Net savings curve](figures/net_savings_curve.png)", "",
          "## Interpretation",
          "- Gross avoidable cost is large, but whether the program pays for itself",
          "  depends mostly on its cost per patient and how well it works.",
          "- Enrolling fewer, higher-risk patients is more cost-efficient: the model's",
          "  value is in deciding who to enroll, not in enrolling more people.",
          "- The simulation assumes the intervention works equally well in every tier;",
          "  real programs should be piloted and measured.", ""]
    report = "\n".join(L)
    (config.REPORTS / "07_savings_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
