"""Step 6: lift-based risk tiering (test set only).

Questions answered:
  - If we target the top X% highest-risk patients, what share of all
    readmissions do we reach? (cumulative gains / capture rate)
  - How much better than random is that? (lift)
  - How stable is the headline number? (bootstrap confidence interval)

Outputs: data/processed/test_tiered.csv, reports/figures/gains_chart.png,
         reports/figures/lift_by_decile.png
Report:  reports/06_tiering_report.md
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.calibrate import assign_tiers

TARGET = "readmit_30"
HEADLINE_PCT = 20          # "top 20% of patients capture X% of readmissions"
N_BOOTSTRAP = 1000


def capture_at(y, score, pct):
    """Share of all readmissions found in the top pct% of scores."""
    n_top = int(round(len(y) * pct / 100))
    order = np.argsort(-score, kind="stable")
    return y[order[:n_top]].sum() / y.sum()


def gains_curve(y, score, steps=100):
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    pcts = np.linspace(0, 100, steps + 1)
    idx = np.clip((pcts / 100 * len(y)).astype(int) - 1, 0, len(y) - 1)
    return pcts, np.where(pcts == 0, 0, cum[idx])


def decile_table(y, score, prob):
    """Deciles ranked by score; 'mean_pred' is the calibrated probability."""
    df = pd.DataFrame({"y": y, "score": score, "prob": prob})
    # rank first so ties don't collapse deciles; decile 1 = highest risk
    df["decile"] = pd.qcut(df["score"].rank(method="first", ascending=False),
                           10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(
        encounters=("y", "size"), readmissions=("y", "sum"),
        mean_pred=("prob", "mean"), actual=("y", "mean"))
    base = y.mean()
    t["lift"] = t["actual"] / base
    t["cum_capture"] = t["readmissions"].cumsum() / y.sum()
    return t


def bootstrap_capture(y, score, pct, n=N_BOOTSTRAP, seed=config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    vals = [capture_at(y[i], score[i], pct)
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def tier_summary(y, score, tiers):
    df = pd.DataFrame({"y": y, "score": score, "tier": tiers})
    t = df.groupby("tier", observed=True).agg(
        encounters=("y", "size"), readmissions=("y", "sum"),
        predicted=("score", "mean"), actual=("y", "mean"))
    t["share_of_patients"] = t["encounters"] / len(df)
    t["share_of_readmissions"] = t["readmissions"] / y.sum()
    t["lift"] = t["actual"] / y.mean()
    return t


# ---------------------------------------------------------------- plots
def plot_gains(y, score, path):
    pcts, model = gains_curve(y, score)
    perfect_x = y.mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(pcts, model * 100, lw=2, label="Model")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random targeting")
    ax.plot([0, perfect_x, 100], [0, 100, 100], ":", color="gray", label="Perfect model")
    cap = capture_at(y, score, HEADLINE_PCT) * 100
    ax.scatter([HEADLINE_PCT], [cap], color="crimson", zorder=5)
    ax.annotate(f"Top {HEADLINE_PCT}% -> {cap:.0f}% of readmissions",
                (HEADLINE_PCT, cap), xytext=(30, cap - 15),
                arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="% of patients targeted (highest risk first)",
           ylabel="% of 30-day readmissions captured",
           title="Cumulative gains (test set)", xlim=(0, 100), ylim=(0, 100))
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_lift(deciles, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(deciles.index.astype(int), deciles["lift"], color="steelblue")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(xlabel="Risk decile (1 = highest risk)", ylabel="Lift vs. average",
           title="Lift by decile (test set)", xticks=range(1, 11))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    preds = pd.read_csv(config.DATA_PROCESSED / "predictions_calibrated.csv")
    test = preds[preds["split"] == "test"].reset_index(drop=True)
    y = test[TARGET].to_numpy()
    # Rank by raw score (continuous, no ties); report calibrated probabilities.
    # See the note in calibrate.py for why.
    rank_score = test["raw_score"].to_numpy()
    prob = test["cal_score"].to_numpy()

    thresholds = joblib.load(config.MODELS / "calibrator.joblib")["tier_thresholds"]
    test["tier"] = assign_tiers(rank_score, thresholds)

    headline = capture_at(y, rank_score, HEADLINE_PCT)
    ci = bootstrap_capture(y, rank_score, HEADLINE_PCT)
    deciles = decile_table(y, rank_score, prob)
    tiers = tier_summary(y, prob, test["tier"])
    top_tiers = tiers.loc[["Very high", "High"]]

    fig_dir = config.REPORTS / "figures"
    plot_gains(y, rank_score, fig_dir / "gains_chart.png")
    plot_lift(deciles, fig_dir / "lift_by_decile.png")
    test.to_csv(config.DATA_PROCESSED / "test_tiered.csv", index=False)

    pct = lambda v: f"{v:.1%}"
    L = ["# Step 6 - Risk Tiering & Lift (test set)", "",
         f"Test set: {len(y):,} encounters, {int(y.sum()):,} readmissions "
         f"(base rate {y.mean():.1%}).", "",
         "## Headline",
         f"Targeting the **top {HEADLINE_PCT}%** highest-risk patients reaches "
         f"**{headline:.1%}** of all 30-day readmissions "
         f"(95% bootstrap CI {ci[0]:.1%}-{ci[1]:.1%}), "
         f"vs. {HEADLINE_PCT}% for random targeting: "
         f"**{headline / (HEADLINE_PCT / 100):.2f}x lift**.", "",
         "## Risk tiers",
         "Cutoffs set on validation data, applied to test.", "",
         "| Tier | Patients | Share of patients | Readmissions | Share of readmissions "
         "| Predicted | Actual | Lift |", "|---|---|---|---|---|---|---|---|"]
    for t, r in tiers.iterrows():
        L.append(f"| {t} | {int(r['encounters']):,} | {pct(r['share_of_patients'])} | "
                 f"{int(r['readmissions']):,} | {pct(r['share_of_readmissions'])} | "
                 f"{pct(r['predicted'])} | {pct(r['actual'])} | {r['lift']:.2f}x |")
    L += ["", f"Very high + High tiers together: {pct(top_tiers['share_of_patients'].sum())} "
          f"of patients, {pct(top_tiers['share_of_readmissions'].sum())} of readmissions.", "",
          "## Decile table",
          "| Decile | Encounters | Readmissions | Predicted | Actual | Lift | Cumulative capture |",
          "|---|---|---|---|---|---|---|"]
    for d, r in deciles.iterrows():
        L.append(f"| {d} | {int(r['encounters']):,} | {int(r['readmissions']):,} | "
                 f"{pct(r['mean_pred'])} | {pct(r['actual'])} | {r['lift']:.2f}x | "
                 f"{pct(r['cum_capture'])} |")
    L += ["", "![Gains chart](figures/gains_chart.png)", "",
          "![Lift by decile](figures/lift_by_decile.png)", "",
          "## Reading this honestly",
          "- The model concentrates risk but does not isolate it: most readmissions",
          "  still come from outside the top 20%. This is typical at AUC ~0.68.",
          "- The top tier is where outreach is most efficient; lower tiers are better",
          "  served by low-cost, population-wide measures.", ""]
    report = "\n".join(L)
    (config.REPORTS / "06_tiering_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
