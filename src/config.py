
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

KEY_COLS = ["region", "crop", "date"]
VOLUME_COL = "volume"
PRICE_COL = "price"

VOLUME_RENAME = {
    "Region": "region",
    "Crop": "crop",
    "Period": "date",
    "Volume_MT": "volume"
}

PRICE_RENAME = {
    "region_name": "region",
    "commodity": "crop",
    "month": "date",
    "farmgate_price_php_kg": "price"
}

VOLUME_RAW_SCHEMA = list(VOLUME_RENAME.keys())
PRICE_RAW_SCHEMA = list(PRICE_RENAME.keys())

ANOMALY_Z_THRESHOLD = 2.5
RANDOM_SEED = 0
