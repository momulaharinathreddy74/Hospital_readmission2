"""Run the full pipeline in order and check key results.

Usage (from the project root):
    python3 run_all.py              # run every step
    python3 run_all.py --from clean # start from a given step
"""
import subprocess
import sys
import time

import pandas as pd

STEPS = ["ingest", "audit", "clean", "leakage", "model", "calibrate", "tiering", "savings"]


def run(step):
    print(f"\n{'=' * 60}\n>>> {step}\n{'=' * 60}")
    t = time.time()
    result = subprocess.run([sys.executable, "-m", f"src.{step}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-3000:])
        sys.exit(f"FAILED at step '{step}'. Fix the error above and rerun.")
    print(f"OK ({time.time() - t:.0f}s)")


def check(name, actual, expected, tol=0.0):
    ok = abs(actual - expected) <= tol
    print(f"  [{'PASS' if ok else 'CHECK'}] {name}: {actual} (expected ~{expected})")
    return ok


def verify():
    print(f"\n{'=' * 60}\n>>> Verifying outputs\n{'=' * 60}")
    from src import config
    clean = pd.read_csv(config.DATA_PROCESSED / "encounters_clean.csv")
    table = pd.read_csv(config.DATA_PROCESSED / "model_table.csv")
    results = [
        check("clean rows", len(clean), 99_340),
        check("readmission rate", round(clean["readmit_30"].mean(), 3), 0.114, 0.002),
        check("leaked feature removed", int("total_encounters" in table), 0),
        check("safe feature kept", int("prior_encounters" in table), 1),
    ]
    preds = pd.read_csv(config.DATA_PROCESSED / "predictions.csv")
    from sklearn.metrics import roc_auc_score
    te = preds[preds["split"] == "test"]
    results.append(check("test ROC-AUC", round(roc_auc_score(te["readmit_30"], te["raw_score"]), 3),
                         0.679, 0.01))
    splits = preds.groupby("patient_nbr")["split"].nunique()
    results.append(check("patients in >1 split", int((splits > 1).sum()), 0))
    results.append(check("model file exists", int((config.MODELS / "gb_model.joblib").exists()), 1))
    cal = pd.read_csv(config.DATA_PROCESSED / "predictions_calibrated.csv")
    cte = cal[cal["split"] == "test"]
    results.append(check("calibrated mean risk (test)", round(cte["cal_score"].mean(), 3),
                         round(cte["readmit_30"].mean(), 3), 0.01))
    tiered = pd.read_csv(config.DATA_PROCESSED / "test_tiered.csv")
    top = tiered["tier"].isin(["Very high", "High"])
    results.append(check("top tiers: share of patients", round(top.mean(), 2), 0.20, 0.02))
    results.append(check("top tiers: share of readmissions",
                         round(tiered.loc[top, "readmit_30"].sum() / tiered["readmit_30"].sum(), 3),
                         0.387, 0.01))
    # tiers must be identical however they are recomputed (guards the tie bug)
    import joblib
    from src.calibrate import assign_tiers
    th = joblib.load(config.MODELS / "calibrator.joblib")["tier_thresholds"]
    recomputed = assign_tiers(tiered["raw_score"].to_numpy(), th).astype(str)
    results.append(check("tier assignment reproducible",
                         int((recomputed != tiered["tier"].astype(str)).sum()), 0))
    # Savings simulation reproduces the resume range
    from src.savings import simulate
    from src.assumptions import value
    top_share = top.mean()
    cap = tiered.loc[top, "readmit_30"].sum() / tiered["readmit_30"].sum()
    for eff, expected in [(0.10, 6.7), (0.20, 13.3)]:
        g = simulate(len(clean), clean["readmit_30"].sum(), top_share, cap, eff,
                     value("cost_per_readmission"), 0)["gross_savings"] / 1e6
        results.append(check(f"gross savings @ {eff:.0%} ($M)", round(g, 1), expected, 0.2))
    for f in ["01_data_audit.md", "02_cleaning_report.md", "03_leakage_report.md",
              "04_model_report.md", "05_calibration_report.md", "06_tiering_report.md",
              "07_savings_report.md"]:
        results.append(check(f"report {f} exists", int((config.REPORTS / f).exists()), 1))
    print("\nAll checks passed." if all(results) else "\nSome checks need a look.")


if __name__ == "__main__":
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0]
    for step in STEPS[STEPS.index(start):]:
        run(step)
    verify()
