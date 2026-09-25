"""Step 3: leakage detection.

Two candidate patient-history features are tested:
  - prior_encounters   : # of this patient's encounters BEFORE this one  (past only)
  - total_encounters   : # of this patient's encounters in the whole dataset
                         (includes FUTURE visits -> suspected leak)

Checks:
  1. Correlation screen  - flags features far more correlated with the target
                           than any legitimate feature.
  2. Point-in-time test  - recompute each feature using only data available at
                           that encounter's discharge; a leak-free feature is
                           unchanged.
  3. Ablation test       - model AUC with vs. without the feature.
  4. Split test          - random row split vs. patient-level split, to show
                           why the model must be split by patient.

Output: data/processed/model_table.csv (clean data + features that passed)
Report: reports/03_leakage_report.md
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src import config
from src.clean import ID_COLS, TARGET

CORR_FLAG_THRESHOLD = 0.20   # far above the strongest legitimate feature (~0.17)
AUC_JUMP_FLAG = 0.03         # a single feature adding this much AUC is suspicious


# ---------------------------------------------------------------- features
def add_patient_history(df):
    """encounter_id is assigned sequentially, so sorting by it gives time order."""
    df = df.sort_values("encounter_id").copy()
    g = df.groupby("patient_nbr")
    df["prior_encounters"] = g.cumcount()
    df["total_encounters"] = g["encounter_id"].transform("count")
    return df


CANDIDATES = ["prior_encounters", "total_encounters"]


# ---------------------------------------------------------------- checks
def correlation_screen(df):
    num = df.drop(columns=ID_COLS + [TARGET]).select_dtypes("number")
    corr = num.corrwith(df[TARGET]).abs().sort_values(ascending=False)
    return corr


def point_in_time_test(df, feature, n_checks=500, seed=config.RANDOM_STATE):
    """For a sample of encounters, recompute the feature using only encounters
    up to and including this one. Returns the share of rows where the value
    in the table differs from the point-in-time value."""
    rng = np.random.default_rng(seed)
    multi = df[df.groupby("patient_nbr")["encounter_id"].transform("count") > 1]
    sample = multi.sample(n=min(n_checks, len(multi)), random_state=seed)
    by_patient = {p: grp for p, grp in multi.groupby("patient_nbr")}
    mismatches = 0
    for _, row in sample.iterrows():
        history = by_patient[row["patient_nbr"]]
        history = history[history["encounter_id"] <= row["encounter_id"]]
        recomputed = add_patient_history(history).set_index("encounter_id")
        if recomputed.loc[row["encounter_id"], feature] != row[feature]:
            mismatches += 1
    return mismatches / len(sample)


def _xy(df, drop):
    X = pd.get_dummies(df.drop(columns=ID_COLS + [TARGET] + drop), dtype=float)
    return X, df[TARGET]


def _fit_auc(X_tr, y_tr, X_te, y_te):
    model = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])


def patient_split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df["patient_nbr"]))
    return df.iloc[tr], df.iloc[te]


def ablation_test(df):
    tr, te = patient_split(df)
    results = {}
    configs = {"baseline (no history features)": CANDIDATES,
               "+ prior_encounters": ["total_encounters"],
               "+ total_encounters": ["prior_encounters"]}
    for name, drop in configs.items():
        X_tr, y_tr = _xy(tr, drop)
        X_te, y_te = _xy(te, drop)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
        results[name] = _fit_auc(X_tr, y_tr, X_te, y_te)
    return results


def split_test(df):
    """Same features, two ways of splitting."""
    X, y = _xy(df, CANDIDATES)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      random_state=config.RANDOM_STATE, stratify=y)
    random_auc = _fit_auc(X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_te])
    overlap = len(set(df["patient_nbr"].iloc[idx_tr]) & set(df["patient_nbr"].iloc[idx_te]))

    tr, te = patient_split(df)
    X_tr, y_tr = _xy(tr, CANDIDATES)
    X_te, y_te = _xy(te, CANDIDATES)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    patient_auc = _fit_auc(X_tr, y_tr, X_te, y_te)
    return random_auc, patient_auc, overlap


# ---------------------------------------------------------------- main
def main():
    df = pd.read_csv(config.DATA_PROCESSED / "encounters_clean.csv")
    df = add_patient_history(df)

    corr = correlation_screen(df)
    top_legit = corr.drop(CANDIDATES).iloc[0]
    pit = {f: point_in_time_test(df, f) for f in CANDIDATES}
    abl = ablation_test(df)
    base = abl["baseline (no history features)"]
    rand_auc, pat_auc, overlap = split_test(df)

    verdict = {}
    for f in CANDIDATES:
        jump = abl[f"+ {f}"] - base
        flags = []
        if corr[f] > CORR_FLAG_THRESHOLD:
            flags.append(f"correlation {corr[f]:.3f} > {CORR_FLAG_THRESHOLD}")
        if pit[f] > 0:
            flags.append(f"value changes for {pit[f]:.0%} of rows when future data is hidden")
        if jump > AUC_JUMP_FLAG:
            flags.append(f"adds {jump:+.3f} AUC on its own")
        verdict[f] = flags

    leaked = [f for f, flags in verdict.items() if flags]
    kept = [f for f in CANDIDATES if f not in leaked]

    L = ["# Step 3 - Leakage Detection", "",
         "## 1. Correlation screen (|r| with 30-day readmission)",
         "| Feature | abs(r) |", "|---|---|"]
    L += [f"| {'**' + f + '**' if f in CANDIDATES else f} | {v:.3f} |"
          for f, v in corr.head(8).items()]
    L += ["", f"Strongest legitimate feature: {corr.drop(CANDIDATES).index[0]} "
          f"({top_legit:.3f}).", "",
          "## 2. Point-in-time test",
          "Each feature is recomputed using only encounters up to the one being",
          "scored (what a hospital would actually know at discharge).", "",
          "| Feature | Rows that change |", "|---|---|"]
    L += [f"| {f} | {pit[f]:.0%} |" for f in CANDIDATES]
    L += ["", "## 3. Ablation test (patient-level split, ROC-AUC)",
          "| Model | AUC | Change |", "|---|---|---|"]
    L += [f"| {k} | {v:.3f} | {v - base:+.3f} |" for k, v in abl.items()]
    L += ["", "## 4. Split test (same features, no history features)",
          "| Split | AUC |", "|---|---|",
          f"| Random row split | {rand_auc:.3f} |",
          f"| Patient-level split | {pat_auc:.3f} |", "",
          f"The random split put {overlap:,} patients in both train and test.", "",
          "## Verdict"]
    for f in CANDIDATES:
        if verdict[f]:
            L.append(f"- **{f}: LEAKED, removed.** " + "; ".join(verdict[f]) + ".")
        else:
            L.append(f"- **{f}: passed, kept.**")
    L += ["", "Why `total_encounters` leaks: a patient readmitted within 30 days",
          "necessarily has another encounter later in the data, so counting all",
          "encounters partially encodes the answer. It is unknowable at discharge.", ""]

    report = "\n".join(L)
    (config.REPORTS / "03_leakage_report.md").write_text(report)
    df.drop(columns=leaked).to_csv(config.DATA_PROCESSED / "model_table.csv", index=False)
    print(report)
    print(f"Kept: {kept}  Removed: {leaked}")


if __name__ == "__main__":
    main()
