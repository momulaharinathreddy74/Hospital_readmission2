"""Central configuration: paths, constants, and random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

RAW_FILE = DATA_RAW / "diabetic_data.csv"
IDS_MAPPING_FILE = DATA_RAW / "IDs_mapping.csv"

# Official UCI source, plus a GitHub mirror as fallback
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
MIRROR_BASE = (
    "https://raw.githubusercontent.com/AkankshaUtreja/"
    "Diabetic-Patients-Readmission-Prediction/master"
)

EXPECTED_SHAPE = (101_766, 50)
NA_VALUES = ["?"]           # this dataset encodes missing values as '?'
TARGET_COL = "readmitted"
POSITIVE_CLASS = "<30"      # 30-day readmission, the CMS HRRP definition
RANDOM_STATE = 42

for p in (DATA_RAW, DATA_PROCESSED, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# The 23 medication columns in the raw data
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
