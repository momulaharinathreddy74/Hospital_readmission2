"""Step 1: download the raw data (if needed) and validate it."""
import io
import sys
import urllib.request
import zipfile

import pandas as pd

from src import config


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def download_raw(force: bool = False) -> None:
    if config.RAW_FILE.exists() and config.IDS_MAPPING_FILE.exists() and not force:
        print(f"Raw data already present at {config.DATA_RAW}")
        return
    try:
        print("Downloading from UCI...")
        with zipfile.ZipFile(io.BytesIO(_download(config.UCI_ZIP_URL))) as z:
            for name in z.namelist():
                if name.endswith(("diabetic_data.csv", "IDs_mapping.csv")):
                    (config.DATA_RAW / name.split("/")[-1]).write_bytes(z.read(name))
    except Exception as e:  # UCI is sometimes slow or blocked
        print(f"UCI download failed ({e}); falling back to mirror...")
        for fname in ("diabetic_data.csv", "IDs_mapping.csv"):
            (config.DATA_RAW / fname).write_bytes(
                _download(f"{config.MIRROR_BASE}/{fname}")
            )


def load_raw() -> pd.DataFrame:
    # keep_default_na=False: in this dataset the string "None" means
    # "test not performed" (max_glu_serum, A1Cresult) and must NOT become NaN.
    df = pd.read_csv(config.RAW_FILE, na_values=config.NA_VALUES,
                     keep_default_na=False, low_memory=False)
    validate_raw(df)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Fail loudly if the file is not the dataset we expect."""
    assert df.shape == config.EXPECTED_SHAPE, f"Unexpected shape {df.shape}"
    assert df["encounter_id"].is_unique, "encounter_id should be unique"
    assert set(df[config.TARGET_COL].unique()) == {"NO", ">30", "<30"}


if __name__ == "__main__":
    download_raw(force="--force" in sys.argv)
    df = load_raw()
    print(f"Loaded and validated: {df.shape[0]:,} rows x {df.shape[1]} cols")
