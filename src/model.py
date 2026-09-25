"""Step 4: train the readmission model.

- Patient-level 60/20/20 split (train / validation / test): no patient appears
  in more than one split.
- Baseline: logistic regression.
- Main model: gradient boosting (sklearn HistGradientBoostingClassifier),
  tuned on the validation set. The test set is scored once, at the end.

Outputs: models/gb_model.joblib, data/processed/predictions.csv
Report:  reports/04_model_report.md
"""
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.clean import ID_COLS, TARGET

# Published results on this dataset for 30-day readmission are roughly 0.64-0.70
BENCHMARK_RANGE = (0.64, 0.70)

PARAM_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [3, 5],
    "max_iter": [200, 400],
    "class_weight": [None, "balanced"],
}


# ---------------------------------------------------------------- data
def load_table():
    df = pd.read_csv(config.DATA_PROCESSED / "model_table.csv")
    cat_cols = df.drop(columns=ID_COLS + [TARGET]).select_dtypes(exclude="number").columns
    df[cat_cols] = df[cat_cols].astype("category")
    return df


def split_by_patient(df):
    """60/20/20 split where every patient lands in exactly one split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    trval_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
    tr_idx, val_idx = next(gss.split(trval, groups=trval["patient_nbr"]))
    train, val = trval.iloc[tr_idx], trval.iloc[val_idx]

    ids = [set(s["patient_nbr"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "patient overlap!"
    return train, val, test


def xy(df):
    return df.drop(columns=ID_COLS + [TARGET]), df[TARGET]


# ---------------------------------------------------------------- models
def make_gb(**params):
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=config.RANDOM_STATE,
        early_stopping=False, **params)


def make_logreg(X):
    cats = X.select_dtypes("category").columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([("num", StandardScaler(), nums),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def metrics(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "mean_pred": p.mean(), "actual_rate": y.mean()}


def tune(X_tr, y_tr, X_val, y_val):
    rows = []
    keys = list(PARAM_GRID)
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, values))
        m = make_gb(**params).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        rows.append({**params, "val_auc": auc})
        print(f"  {params} -> val AUC {auc:.4f}")
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False)


# ---------------------------------------------------------------- main
def main():
    df = load_table()
    train, val, test = split_by_patient(df)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = map(xy, (train, val, test))

    print("Baseline: logistic regression")
    logreg = make_logreg(X_tr).fit(X_tr, y_tr)
    lr_val_auc = roc_auc_score(y_val, logreg.predict_proba(X_val)[:, 1])

    print("Tuning gradient boosting on validation set...")
    grid = tune(X_tr, y_tr, X_val, y_val)
    best = grid.iloc[0].drop("val_auc").to_dict()
    best = {k: (None if pd.isna(v) else v) for k, v in best.items()}
    best["max_iter"], best["max_depth"] = int(best["max_iter"]), int(best["max_depth"])

    model = make_gb(**best).fit(X_tr, y_tr)

    # Final, one-time evaluation on the untouched test set
    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    m_val, m_te = metrics(y_val, p_val), metrics(y_te, p_te)
    m_lr_te = metrics(y_te, logreg.predict_proba(X_te)[:, 1])

    print("Permutation importance (validation sample)...")
    samp = val.sample(n=min(8000, len(val)), random_state=config.RANDOM_STATE)
    Xs, ys = xy(samp)
    imp = permutation_importance(model, Xs, ys, scoring="roc_auc", n_repeats=5,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=Xs.columns).sort_values(ascending=False)

    # Save model + predictions for Steps 5-7
    joblib.dump({"model": model, "params": best, "features": list(X_tr.columns)},
                config.MODELS / "gb_model.joblib")
    preds = pd.concat([
        s[ID_COLS + [TARGET, "specialty"]].assign(split=name, raw_score=p)
        for name, s, p in [("train", train, model.predict_proba(X_tr)[:, 1]),
                           ("val", val, p_val), ("test", test, p_te)]])
    preds.to_csv(config.DATA_PROCESSED / "predictions.csv", index=False)

    lo, hi = BENCHMARK_RANGE
    in_range = lo <= m_te["roc_auc"] <= hi
    L = ["# Step 4 - Readmission Model", "",
         "## Patient-level split",
         "| Split | Encounters | Patients | Readmit rate |", "|---|---|---|---|"]
    L += [f"| {n} | {len(s):,} | {s['patient_nbr'].nunique():,} | {s[TARGET].mean():.1%} |"
          for n, s in [("train", train), ("validation", val), ("test", test)]]
    L += ["", "No patient appears in more than one split (asserted in code).", "",
          "## Hyperparameter search (top 5 of "
          f"{len(grid)}, scored on validation)",
          "| learning_rate | max_depth | max_iter | class_weight | val AUC |",
          "|---|---|---|---|---|",
          *[f"| {r.learning_rate} | {r.max_depth} | {r.max_iter} | "
            f"{r.class_weight if isinstance(r.class_weight, str) else 'None'} | {r.val_auc:.4f} |"
            for r in grid.head(5).itertuples()], "",
          f"Chosen: `{best}`", "",
          "## Results",
          "| Model | Split | ROC-AUC | PR-AUC | Brier |", "|---|---|---|---|---|",
          f"| Logistic regression | validation | {lr_val_auc:.3f} | | |",
          f"| Logistic regression | test | {m_lr_te['roc_auc']:.3f} | "
          f"{m_lr_te['pr_auc']:.3f} | {m_lr_te['brier']:.4f} |",
          f"| Gradient boosting | validation | {m_val['roc_auc']:.3f} | "
          f"{m_val['pr_auc']:.3f} | {m_val['brier']:.4f} |",
          f"| **Gradient boosting** | **test** | **{m_te['roc_auc']:.3f}** | "
          f"{m_te['pr_auc']:.3f} | {m_te['brier']:.4f} |", "",
          f"Test ROC-AUC {m_te['roc_auc']:.3f} is "
          f"{'within' if in_range else 'outside'} the published range ({lo}-{hi}).",
          f"PR-AUC baseline (random model) = prevalence = {m_te['actual_rate']:.3f}.", "",
          "## Probability sanity check (preview of Step 5)",
          f"- Mean predicted risk on test: **{m_te['mean_pred']:.1%}**",
          f"- Actual readmission rate on test: **{m_te['actual_rate']:.1%}**", "",
          "## Top 10 features (permutation importance, drop in ROC-AUC)",
          "| Feature | Importance |", "|---|---|"]
    L += [f"| {f} | {v:.4f} |" for f, v in importance.head(10).items()]
    report = "\n".join(L)
    (config.REPORTS / "04_model_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
