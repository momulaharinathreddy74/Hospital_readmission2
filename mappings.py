"""Code mappings: CMS/UB-04 admission & discharge IDs, and ICD-9 groups.

The raw data stores admission type, admission source and discharge
disposition as numeric IDs whose meanings follow UB-04 claim-form codes
(described in IDs_mapping.csv). We collapse them into a few
clinically meaningful groups.
"""
import pandas as pd

from src import config

# ---------- Admission type (UB-04 form locator 14) ----------
ADMISSION_TYPE_GROUP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Other", 7: "Emergency",
    5: "Unknown", 6: "Unknown", 8: "Unknown",
}

# ---------- Discharge disposition (UB-04 form locator 17) ----------
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]
DISCHARGE_GROUP = {
    1: "Home",
    6: "Home_with_services", 8: "Home_with_services",
    3: "Facility_SNF_LTC", 4: "Facility_SNF_LTC", 15: "Facility_SNF_LTC",
    22: "Facility_SNF_LTC", 23: "Facility_SNF_LTC", 24: "Facility_SNF_LTC",
    2: "Transfer_hospital", 5: "Transfer_hospital", 9: "Transfer_hospital",
    10: "Transfer_hospital", 27: "Transfer_hospital", 28: "Transfer_hospital",
    29: "Transfer_hospital", 30: "Transfer_hospital",
    7: "Left_AMA",
    12: "Outpatient_followup", 16: "Outpatient_followup", 17: "Outpatient_followup",
    18: "Unknown", 25: "Unknown", 26: "Unknown",
}

# ---------- Admission source (UB-04 form locator 15) ----------
ADMISSION_SOURCE_GROUP = {
    1: "Referral", 2: "Referral", 3: "Referral",
    7: "Emergency_room",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
    18: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer",
    8: "Other", 11: "Other", 12: "Other", 13: "Other", 14: "Other",
    19: "Other", 23: "Other", 24: "Other",
    9: "Unknown", 15: "Unknown", 17: "Unknown", 20: "Unknown", 21: "Unknown",
}


def load_id_descriptions() -> dict[str, pd.Series]:
    """Parse IDs_mapping.csv (three tables stacked in one file, separated by
    blank lines) into {id_column: Series(id -> description)}."""
    tables, current, rows = {}, None, []
    for line in config.IDS_MAPPING_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line == ",":
            continue
        key, _, desc = line.partition(",")
        if key.endswith("_id"):
            if current:
                tables[current] = pd.Series(dict(rows))
            current, rows = key, []
        else:
            rows.append((int(key), desc.strip('"')))
    tables[current] = pd.Series(dict(rows))
    return tables


# ---------- ICD-9 grouping (Strack et al., 2014, the dataset's source paper) ----------
def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"               # supplementary / external-cause codes
    if code.startswith("250"):
        return "Diabetes"
    n = float(code)
    if 390 <= n <= 459 or int(n) == 785:
        return "Circulatory"
    if 460 <= n <= 519 or int(n) == 786:
        return "Respiratory"
    if 520 <= n <= 579 or int(n) == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or int(n) == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"
