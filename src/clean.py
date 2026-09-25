"""Step 2: clean the raw data and engineer features.

Output: data/processed/encounters_clean.csv
Report: reports/02_cleaning_report.md

Deliberately NOT done here: any feature built by grouping on patient_nbr.
Those are leakage risks and are tested separately in Step 3.
"""
import pandas as pd

from src import config
from src.ingest import load_raw
from src.mappings import (ADMISSION_SOURCE_GROUP, ADMISSION_TYPE_GROUP,
                          DISCHARGE_GROUP, EXPIRED_OR_HOSPICE, icd9_group)

ID_COLS = ["encounter_id", "patient_nbr"]
TARGET = "readmit_30"
TOP_SPECIALTIES = 12          # keep the largest departments, pool the rest
MIN_DRUG_USAGE = 0.01         # drop drug columns used in <1% of encounters

AGE_MIDPOINT = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}


class CleaningLog:
    """Records row counts after each filter, for the report."""
    def __init__(self, df):
        self.steps = [("Raw data", len(df))]

    def add(self, name, df):
        self.steps.append((name, len(df)))

    def to_markdown(self):
        lines = ["| Step | Rows | Removed |", "|---|---|---|"]
        prev = None
        for name, n in self.steps:
            removed = "" if prev is None else f"{prev - n:,}"
            lines.append(f"| {name} | {n:,} | {removed} |")
            prev = n
        return "\n".join(lines)


def check_mappings_complete(df):
    """Every ID present in the data must have an explicit group, so nothing is
    silently lumped into 'Unknown' by a missing dictionary entry."""
    for col, mapping in [("admission_type_id", ADMISSION_TYPE_GROUP),
                         ("discharge_disposition_id", DISCHARGE_GROUP),
                         ("admission_source_id", ADMISSION_SOURCE_GROUP)]:
        unmapped = set(df[col].dropna().unique()) - set(mapping)
        assert not unmapped, f"{col} has unmapped IDs: {sorted(unmapped)}"


def apply_exclusions(df, log):
    # Patients who died or went to hospice can't meaningfully be readmitted.
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    log.add("Exclude expired / hospice discharges", df)
    df = df[df["gender"] != "Unknown/Invalid"]
    log.add("Exclude invalid gender", df)
    return df.copy()


def engineer_features(df, drug_cols_kept):
    out = pd.DataFrame(index=df.index)
    out[ID_COLS] = df[ID_COLS]

    # --- Demographics ---
    out["age_years"] = df["age"].map(AGE_MIDPOINT)
    out["gender"] = df["gender"]
    out["race"] = df["race"].fillna("Unknown")

    # --- CMS / UB-04 code groupings ---
    out["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_GROUP).fillna("Unknown")
    out["discharge_group"] = df["discharge_disposition_id"].map(DISCHARGE_GROUP).fillna("Unknown")
    out["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_GROUP).fillna("Unknown")

    # --- Department (medical specialty) and payer ---
    spec = df["medical_specialty"].fillna("Unknown")
    top = spec[spec != "Unknown"].value_counts().nlargest(TOP_SPECIALTIES).index
    out["specialty"] = spec.where(spec.isin(top) | (spec == "Unknown"), "Other")
    out["payer_code"] = df["payer_code"].fillna("Unknown")

    # --- Encounter intensity ---
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures",
              "num_medications", "number_diagnoses"]:
        out[c] = df[c]

    # --- Prior utilization (counts from the year BEFORE this encounter: safe) ---
    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        out[c] = df[c]
    out["total_prior_visits"] = out[["number_outpatient", "number_emergency",
                                     "number_inpatient"]].sum(axis=1)

    # --- ICD-9 diagnoses -> clinical groups ---
    for c in ["diag_1", "diag_2", "diag_3"]:
        out[f"{c}_group"] = df[c].map(icd9_group)
    out["has_diabetes_primary"] = (out["diag_1_group"] == "Diabetes").astype(int)

    # --- Lab results ("None" = test not performed, a real category) ---
    out["A1Cresult"] = df["A1Cresult"]
    out["max_glu_serum"] = df["max_glu_serum"]
    out["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)

    # --- Medications ---
    for c in drug_cols_kept:
        out[f"med_{c}"] = df[c]
    all_drugs = [c for c in config.DRUG_COLS]
    out["n_meds_prescribed"] = (df[all_drugs] != "No").sum(axis=1)
    out["n_meds_changed"] = df[all_drugs].isin(["Up", "Down"]).sum(axis=1)
    out["med_change"] = (df["change"] == "Ch").astype(int)
    out["diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    # --- Target ---
    out[TARGET] = (df["readmitted"] == config.POSITIVE_CLASS).astype(int)
    return out


def validate(df):
    assert df["encounter_id"].is_unique
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    nulls = df[feature_cols].isna().sum()
    assert nulls.sum() == 0, f"Unexpected nulls:\n{nulls[nulls > 0]}"
    assert df[TARGET].isin([0, 1]).all()


def build_report(log, clean, dropped_cols):
    feats = [c for c in clean.columns if c not in ID_COLS + [TARGET]]
    lines = ["# Step 2 - Cleaning & Feature Engineering", "",
             "## Row filtering log", log.to_markdown(), "",
             f"Final: **{len(clean):,} encounters**, "
             f"{clean['patient_nbr'].nunique():,} patients, "
             f"30-day readmission rate **{clean[TARGET].mean():.1%}**.", "",
             "## Dropped raw columns",
             *[f"- `{c}`: {why}" for c, why in dropped_cols.items()], "",
             f"## Features ({len(feats)})", ", ".join(f"`{c}`" for c in feats), "",
             "## 30-day readmission rate by group", ""]
    for col in ["discharge_group", "admission_type", "diag_1_group", "specialty"]:
        t = (clean.groupby(col)[TARGET].agg(["count", "mean"])
             .sort_values("mean", ascending=False))
        lines += [f"### {col}", "| Group | Encounters | Readmit rate |", "|---|---|---|"]
        lines += [f"| {g} | {int(r['count']):,} | {r['mean']:.1%} |" for g, r in t.iterrows()]
        lines.append("")
    return "\n".join(lines)


def main():
    raw = load_raw()
    log = CleaningLog(raw)
    df = apply_exclusions(raw, log)
    check_mappings_complete(df)

    usage = (df[config.DRUG_COLS] != "No").mean()
    drug_cols_kept = usage[usage >= MIN_DRUG_USAGE].index.tolist()

    clean = engineer_features(df, drug_cols_kept)
    validate(clean)

    dropped = {"weight": "96.9% missing",
               "medical_specialty": "replaced by grouped `specialty`",
               "admission_type_id / discharge_disposition_id / admission_source_id":
                   "replaced by UB-04 groupings",
               "diag_1/2/3": "replaced by ICD-9 groups",
               "readmitted": "replaced by binary `readmit_30`"}
    for c in config.DRUG_COLS:
        if c not in drug_cols_kept:
            dropped[c] = f"used in {usage[c]:.2%} of encounters (kept in med counts)"

    out = config.DATA_PROCESSED / "encounters_clean.csv"
    clean.to_csv(out, index=False)
    report = build_report(log, clean, dropped)
    (config.REPORTS / "02_cleaning_report.md").write_text(report)
    print(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
