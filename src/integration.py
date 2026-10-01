
import logging

import numpy as np
import pandas as pd

from src import config
from src.validation import check_required_columns, check_unique_key

logger = logging.getLogger(__name__)


class IntegrationError(RuntimeError):


    def merge_volume_price(volume, price, keys=config.KEY_COLS):
        check_required_columns(volume, list(keys) + [config.VOLUME_COL], "volume")
        check_required_columns(price, list(keys) + [config.PRICE_COL], "price")
        check_unique_key(volume, keys, "volume")
        check_unique_key(price, keys, "price")

        merged = volume.merge(price, on=keys, how="outer", validate="one_to_one", indicator=True)
        merged["match_status"] = merged.pop("_merge").map(
            {"both": "matched", "left_only": "volume_only", "right_only": "price_only"}).astype(str)
        merged = merged.sort_values(keys).reset_index(drop=True)

        counts = merged["match_status"].value_counts()
        n_matched = int(counts.get("matched", 0))
        n_vol_only = int(counts.get("volume_only", 0))
        n_price_only = int(counts.get("price_only", 0))

        report = {
            "n_volume": len(volume), "n_price": len(price), "n_merged": len(merged),
            "matched": n_matched, "volume_only": n_vol_only, "price_only": n_price_only,
            "cardinality": "one_to_one (validated)",
            "volume_total_before": float(volume[config.VOLUME_COL].sum()),
            "volume_total_after": float(merged[config.VOLUME_COL].sum()),
            "price_total_before": float(price[config.PRICE_COL].sum()),
            "price_total_after": float(merged[config.PRICE_COL].sum()),
        }

        problems = []
        if report["n_merged"] != n_matched + n_vol_only + n_price_only:
            problems.append("row count != matched + unmatched")
        if report["n_merged"] != report["n_volume"] + report["n_price"] - n_matched:
            problems.append("row count != n_volume + n_price - matched")
        if n_vol_only != report["n_volume"] - n_matched or n_price_only != report["n_price"] - n_matched:
            problems.append("unmatched counts inconsistent with input sizes")
        if not np.isclose(report["volume_total_before"], report["volume_total_after"]):
            problems.append("volume total changed by merge")
        if not np.isclose(report["price_total_before"], report["price_total_after"]):
            problems.append("price total changed by merge")

        if problems:
            raise IntegrationError("; ".join(problems))
        if n_matched == 0:
            raise IntegrationError("no records matched on keys - check label/date standardisation")

        logger.info("Merged: %d matched, %d volume-only, %d price-only", n_matched, n_vol_only, n_price_only)
        return merged, report


    def unmatched_summary(merged):
        unmatched = merged[merged["match_status"] != "matched"]
        counts = unmatched.groupby(["match_status", "region", "crop"]).size()
        return counts.rename("rows").reset_index().sort_values("rows", ascending=False)


    def analysis_ready(merged):
        """only the rows that matched and have both a volume and a price"""
        mask = (merged["match_status"] == "matched") & merged[config.VOLUME_COL].notna() \
            & merged[config.PRICE_COL].notna()
        return merged.loc[mask].drop(columns="match_status").reset_index(drop=True)
