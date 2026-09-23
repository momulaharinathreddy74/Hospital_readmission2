"""Step 1: data-quality audit of the raw dataset -> reports/01_data_audit.md"""
import pandas as pd

from src import config
from src.ingest import load_raw

# Discharge disposition IDs meaning the patient died or went to hospice.
# These patients cannot (or are unlikely to) be readmitted, so they will be
# excluded in Step 2. Flagging them here is part of the audit.
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]


def audit(df: pd.DataFrame) -> str:
    n = len(df)
    y = df[config.TARGET_COL]
    lines = ["# Step 1 - Raw Data Audit", ""]

    lines += ["## Overview",
              f"- Encounters: {n:,}",
              f"- Unique patients: {df['patient_nbr'].nunique():,}",
              f"- Columns: {df.shape[1]}",
              f"- Patients with >1 encounter: "
              f"{(df['patient_nbr'].value_counts() > 1).sum():,}", ""]

    lines += ["## Target distribution (`readmitted`)", "| Value | Count | % |", "|---|---|---|"]
    for k, v in y.value_counts().items():
        lines.append(f"| {k} | {v:,} | {v / n:.1%} |")
    pos = (y == config.POSITIVE_CLASS).mean()
    lines += ["", f"Binary target = `<30`: positive rate **{pos:.1%}** (imbalanced).", ""]

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    lines += ["## Missing values", "| Column | % missing |", "|---|---|"]
    lines += [f"| {c} | {p:.1%} |" for c, p in miss.items()]
    lines.append("")

    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    near_const = [c for c in df.columns if c not in const
                  and df[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.999]
    lines += ["## Constant / near-constant columns",
              f"- Constant: {const or 'none'}",
              f"- >99.9% single value: {near_const or 'none'}", ""]

    exp = df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)
    lines += ["## Clinical exclusions to handle in Step 2",
              f"- Expired/hospice discharges: {exp.sum():,} encounters "
              f"({exp.mean():.1%}); 30-day readmission rate among them: "
              f"{(y[exp] == '<30').mean():.1%}",
              f"- `gender` = Unknown/Invalid: {(df['gender'] == 'Unknown/Invalid').sum()}",
              ""]

    lines += ["## Identifier columns (must never be model features)",
              "- `encounter_id`, `patient_nbr`: IDs. `patient_nbr` is needed for a",
              "  patient-level train/test split (Step 4), and per-patient aggregates",
              "  are a leakage risk (Step 3).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_raw()
    report = audit(df)
    out = config.REPORTS / "01_data_audit.md"
    out.write_text(report)
    print(report)
    print(f"\nSaved -> {out}")
