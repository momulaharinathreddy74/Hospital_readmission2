"""Step 5: probability calibration.

Problem: the model was trained with class_weight="balanced", which up-weights
readmissions ~8x. Ranking (AUC) is unaffected, but predicted probabilities are
inflated far above the true readmission rate.

Fix: isotonic regression fitted on the VALIDATION set, mapping raw scores to
observed readmission frequencies. Platt (sigmoid) scaling is fitted as a
comparison. The test set is used only for evaluation.

Outputs: models/calibrator.joblib, data/processed/predictions_calibrated.csv,
         reports/figures/reliability_curve.png
Report:  reports/05_calibration_report.md
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config

TARGET = "readmit_30"
N_BINS = 10


# ---------------------------------------------------------------- metrics
def expected_calibration_error(y, p, n_bins=N_BINS):
    """Weighted average gap between predicted and actual rate, over quantile bins."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    g = df.groupby("bin").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return (g["n"] * (g["pred"] - g["actual"]).abs()).sum() / g["n"].sum()


def summarize(y, p):
    return {"ROC-AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p),
            "ECE": expected_calibration_error(y, p), "Mean predicted": p.mean(),
            "Actual rate": y.mean()}


# ---------------------------------------------------------------- tiers
# NOTE: tiers are defined on the RAW model score, not the calibrated score.
# Isotonic output is a step function, so thousands of patients share the exact
# same calibrated value; percentile cutoffs on it are ambiguous and flip with
# tiny floating-point changes. Isotonic is monotone, so ranking by raw score
# gives the same order without ties. Calibrated scores are used for the
# probabilities we report.
def tier_thresholds(scores):
    """Raw-score cutoffs for each tier, computed from a reference set (validation)."""
    return {name: np.percentile(scores, 100 - pct) for name, pct in config.RISK_TIERS}


def assign_tiers(scores, thresholds):
    names = [n for n, _ in config.RISK_TIERS]
    tiers = np.full(len(scores), names[-1], dtype=object)
    for name in reversed(names[:-1]):          # highest tier wins
        tiers[scores >= thresholds[name]] = name
    return pd.Categorical(tiers, categories=names, ordered=True)


def tier_table(y, raw, cal, tiers):
    df = pd.DataFrame({"y": y, "raw": raw, "cal": cal, "tier": tiers})
    return (df.groupby("tier", observed=True)
              .agg(encounters=("y", "size"), raw_pred=("raw", "mean"),
                   calibrated_pred=("cal", "mean"), actual=("y", "mean")))


# ---------------------------------------------------------------- plot
def reliability_plot(y, curves, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for label, p in curves.items():
        bins = pd.qcut(p, N_BINS, labels=False, duplicates="drop")
        d = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b").mean()
        ax.plot(d["p"], d["y"], "o-", label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed readmission rate",
           title="Reliability curve (test set)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    preds = pd.read_csv(config.DATA_PROCESSED / "predictions.csv")
    val = preds[preds["split"] == "val"]
    test = preds[preds["split"] == "test"]

    # Fit calibrators on validation only
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(val["raw_score"], val[TARGET])
    platt = LogisticRegression().fit(val[["raw_score"]], val[TARGET])

    y = test[TARGET].to_numpy()
    raw = test["raw_score"].to_numpy()
    p_iso = iso.predict(raw)
    p_platt = platt.predict_proba(test[["raw_score"]])[:, 1]

    results = pd.DataFrame({"Uncalibrated": summarize(y, raw),
                            "Platt (sigmoid)": summarize(y, p_platt),
                            "Isotonic": summarize(y, p_iso)}).T

    # Tier cutoffs from validation calibrated scores, applied to test
    thresholds = tier_thresholds(val["raw_score"].to_numpy())
    tiers = assign_tiers(raw, thresholds)
    tiers_df = tier_table(y, raw, p_iso, tiers)

    fig_path = config.REPORTS / "figures" / "reliability_curve.png"
    reliability_plot(y, {"Uncalibrated": raw, "Isotonic": p_iso}, fig_path)

    joblib.dump({"calibrator": iso, "tier_thresholds": thresholds},
                config.MODELS / "calibrator.joblib")
    preds["cal_score"] = iso.predict(preds["raw_score"])
    preds.to_csv(config.DATA_PROCESSED / "predictions_calibrated.csv", index=False)

    fmt = lambda v: f"{v:.1%}"
    L = ["# Step 5 - Probability Calibration", "",
         "## The bug",
         "The model was trained with `class_weight=\"balanced\"`, which up-weights",
         "readmissions so the model focuses on them. AUC only measures ranking, so it",
         "could not detect the side effect: every predicted probability is inflated.", "",
         "## Test-set results (calibrators fitted on validation only)",
         "| Method | ROC-AUC | Brier | ECE | Mean predicted | Actual rate |",
         "|---|---|---|---|---|---|"]
    for name, r in results.iterrows():
        L.append(f"| {name} | {r['ROC-AUC']:.3f} | {r['Brier']:.4f} | {r['ECE']:.4f} | "
                 f"{fmt(r['Mean predicted'])} | {fmt(r['Actual rate'])} |")
    L += ["", "ECE = expected calibration error: average gap between predicted and",
          "actual rates across 10 bins (lower is better; 0 = perfect).", "",
          "## Per-tier check (test set)",
          "Tier cutoffs were set on the validation set, then applied to test.", "",
          "| Tier | Encounters | Raw predicted | Calibrated predicted | Actual |",
          "|---|---|---|---|---|"]
    for t, r in tiers_df.iterrows():
        L.append(f"| {t} | {int(r['encounters']):,} | {fmt(r['raw_pred'])} | "
                 f"{fmt(r['calibrated_pred'])} | {fmt(r['actual'])} |")
    L += ["", "![Reliability curve](figures/reliability_curve.png)", "",
          "## Why isotonic",
          "Isotonic regression is non-parametric (it only assumes higher score ->",
          "higher risk), so it can fix any monotone distortion. Platt scaling assumes",
          "a sigmoid shape. With ~20k validation rows there is enough data for",
          "isotonic without overfitting.", ""]
    report = "\n".join(L)
    (config.REPORTS / "05_calibration_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
