
from pathlib import Path

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
