
import logging
import timeit
import warnings

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)
SERIES = ["region", "crop"]



def add_derived(df):
    
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["value_php_m"] = out[config.VOLUME_COL] * 1000 * out[config.PRICE_COL] / 1e6
    grp = out.groupby(SERIES)
    out["vol_ratio"] = out[config.VOLUME_COL] / grp[config.VOLUME_COL].transform("mean")
    out["price_ratio"] = out[config.PRICE_COL] / grp[config.PRICE_COL].transform("mean")
    return out



def value_by_crop(df):
    """total volume, average price and total value for each crop"""
    result = df.groupby("crop").agg(total_volume_mt=(config.VOLUME_COL, "sum"),
                                    mean_price_php_kg=(config.PRICE_COL, "mean"),
                                    total_value_php_m=("value_php_m", "sum"),
                                    n_obs=("value_php_m", "size"))
    return result.sort_values("total_value_php_m", ascending=False)


def value_by_region(df):
    """total value (PHP million) for each region (rows) and crop (columns)"""
    pivot = df.pivot_table(index="region", columns="crop", values="value_php_m", aggfunc="sum", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)



def volume_price_correlation(df):
   
    rows = []
    for crop, g in df.groupby("crop"):
        x = np.log(g["vol_ratio"])
        y = np.log(g["price_ratio"])
        rows.append({"crop": crop, "n_obs": len(g), "spearman_r": x.corr(y, method="spearman")})
    return pd.DataFrame(rows).sort_values("spearman_r").reset_index(drop=True)



def seasonal_price_index(df):
   
    d = df.copy()
    d["annual_mean"] = d.groupby(SERIES + ["year"])[config.PRICE_COL].transform("mean")
    d["price_index"] = d[config.PRICE_COL] / d["annual_mean"] * 100
    return d.groupby(["crop", "month"], as_index=False)["price_index"].mean()



def regional_price_premium(df):
    """percent difference between a region's average price and the crop's overall average price"""
    crop_mean = df.groupby("crop")[config.PRICE_COL].mean().rename("crop_mean")
    rc = df.groupby(["region", "crop"])[config.PRICE_COL].mean().rename("region_mean").reset_index()
    rc = rc.merge(crop_mean, on="crop")
    rc["premium_pct"] = (rc["region_mean"] / rc["crop_mean"] - 1) * 100
    return rc



def build_price_cube(df, value=config.PRICE_COL):
    
    crops = pd.Index(sorted(df["crop"].unique()))
    regions = pd.Index(sorted(df["region"].unique()))
    months = pd.Index(sorted(df["date"].unique()))

    cube = np.full((len(crops), len(regions), len(months)), np.nan)
    crop_idx = crops.get_indexer(df["crop"])
    region_idx = regions.get_indexer(df["region"])
    month_idx = months.get_indexer(df["date"])
    cube[crop_idx, region_idx, month_idx] = df[value].to_numpy()
    return cube, crops, regions, months


def zscore_along(cube, axis=-1):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  
        mean = np.nanmean(cube, axis=axis, keepdims=True)
        std = np.nanstd(cube, axis=axis, keepdims=True)
    return (cube - mean) / np.where(std > 0, std, np.inf)


def zscore_loop(cube):
    out = np.full(cube.shape, np.nan)
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            s = cube[i, j]
            ok = ~np.isnan(s)
            if not ok.any():
                continue
            sd = s[ok].std()
            out[i, j, ok] = (s[ok] - s[ok].mean()) / sd if sd > 0 else 0.0
    return out


def flag_price_anomalies(df, threshold=config.ANOMALY_Z_THRESHOLD):
    
    cube, crops, regions, months = build_price_cube(df)
    z = zscore_along(cube)
    mask = np.abs(np.nan_to_num(z, nan=0.0)) > threshold
    ci, ri, mi = np.nonzero(mask)
    out = pd.DataFrame({"crop": crops[ci], "region": regions[ri], "date": months[mi],
                        "price_php_kg": cube[ci, ri, mi], "z_score": z[ci, ri, mi]})
    return out.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)


def verify_zscore(cube):
    return bool(np.allclose(zscore_along(cube), zscore_loop(cube), equal_nan=True))



def _zscore_pandas(arr2d):
    n, m = arr2d.shape
    long = pd.DataFrame({"s": np.repeat(np.arange(n), m), "x": arr2d.ravel()})
    g = long.groupby("s")["x"]
    mean, std = g.transform("mean"), g.transform("std", ddof=0)
    z = (long["x"] - mean) / np.where(std > 0, std, np.inf)
    return z.to_numpy().reshape(n, m)


def benchmark_zscore(n_series_list=(100, 1_000, 10_000), n_months=60, repeats=5, seed=config.RANDOM_SEED):
   
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_series_list:
        data = rng.normal(50, 10, size=(n, n_months))
        data[rng.random(data.shape) < 0.03] = np.nan
        cube = data[:, None, :]  

        methods = {
            "python_loop": lambda: zscore_loop(cube)[:, 0, :],
            "numpy_vectorised": lambda: zscore_along(cube)[:, 0, :],
            "pandas_groupby": lambda: _zscore_pandas(data),
        }
        reference = methods["python_loop"]()
        base = None
        for name, fn in methods.items():
            times = timeit.repeat(fn, number=1, repeat=repeats)
            ok = bool(np.allclose(fn(), reference, equal_nan=True))
            best = min(times)
            if name == "python_loop":
                base = best
            rows.append({"n_series": n, "n_months": n_months, "n_values": n * n_months, "method": name,
                         "best_s": best, "median_s": float(np.median(times)),
                         "speedup_vs_loop": base / best, "matches_loop": ok})
    return pd.DataFrame(rows)


def benchmark_conclusion(bench, actual_series):
    big = bench[bench["method"] == "numpy_vectorised"].sort_values("n_series").iloc[-1]
    small = bench[(bench["method"] == "python_loop") & (bench["n_series"] == bench["n_series"].min())].iloc[0]
    all_ok = bool(bench["matches_loop"].all())
    return (
        f"All methods matched the loop reference: {all_ok}. At {int(big.n_series):,} series x "
        f"{int(big.n_months)} months NumPy was {big.speedup_vs_loop:,.0f}x faster than the loop. "
        f"This project's real data has {actual_series} region-crop series; even the loop takes about "
        f"{small.best_s * 1000:.1f} ms for {int(small.n_series)} series, so the optimization is not needed "
        "for speed. It is still needed as the default because it is shorter, equally readable, "
        "and keeps working if the dataset grows to many more regions/crops/markets. "
        "Limitations: one machine, synthetic normal data, best-of-N wall-clock timing, "
        "and a single operation (z-score), which in turn, results do not generalize to other computations."
    )
