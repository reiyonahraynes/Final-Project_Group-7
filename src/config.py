
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"

VOLUME_FILE = "crop_volume.csv"
PRICE_FILE = "crop_prices.csv"
MANIFEST_FILE = "raw_manifest.json"

NA_VALUES = ["", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "--"]

VOLUME_RAW_SCHEMA = {
    "Region": "text",
    "Crop": "text",
    "Period": "date",
    "Volume_MT": "numeric",
}
PRICE_RAW_SCHEMA = {
    "region_name": "text",
    "commodity": "text",
    "month": "date",
    "farmgate_price_php_kg": "numeric",
}

MIN_VALID_SHARE = 0.80

VOLUME_RENAME = {"Region": "region", "Crop": "crop", "Period": "date", "Volume_MT": "volume_mt"}
PRICE_RENAME = {
    "region_name": "region",
    "commodity": "crop",
    "month": "date",
    "farmgate_price_php_kg": "price_php_kg",
}


KEY_COLS = ["region", "crop", "date"]
VOLUME_COL = "volume_mt"
PRICE_COL = "price_php_kg"

DATE_FORMATS = ["%Y-%m-%d", "%Y-%m", "%b %Y", "%B %Y", "%m/%Y"]

REGION_VARIANTS = {
    "NCR": ["national capital region", "metro manila"],
    "CALABARZON": ["region iv-a", "region iv-a (calabarzon)", "region 4a"],
    "Central Luzon": ["region iii", "region iii - central luzon"],
    "Western Visayas": ["region vi", "region vi - western visayas"],
    "Central Visayas": ["region vii", "region vii - central visayas"],
    "Northern Mindanao": ["region x", "region x - northern mindanao"],
    "Davao Region": ["region xi", "region xi - davao region", "davao"],
    "Cagayan Valley": ["region ii", "region ii - cagayan valley"],
    "Ilocos Region": ["region i", "region i - ilocos region", "ilocos"],
    "BARMM": ["bangsamoro autonomous region in muslim mindanao"],

CROP_VARIANTS = {
    "Rice": ["palay", "rice (regular milled)"],
    "Corn": ["maize", "corn grain"],
    "Tomato": ["tomatoes"],
    "Onion": ["onions", "red onion"],
    "Cabbage": ["cabbages"],
    "Banana": ["bananas", "saging"],
    "Garlic": ["garlic"],
}


ANOMALY_Z_THRESHOLD = 2.5
RANDOM_SEED = 42
