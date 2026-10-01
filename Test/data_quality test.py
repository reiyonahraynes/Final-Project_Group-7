
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import acquisition, analysis, cleaning, config, integration, run_pipeline, validation  # noqa: E402
from src.validation import DataQualityError, SchemaError  # noqa: E402


@pytest.fixture
def raw_volume():
    return pd.DataFrame({
        "Region": ["NCR", "ncr ", "Region III - Central Luzon", "BARMM", "NCR"],
        "Crop": ["Rice", "RICE", "Palay", "Rice", "Rice"],
        "Period": ["2022-01", "Jan 2022", "01/2022", "2022-01-01", "not a date"],
        "Volume_MT": ["1,200.5", "1200.5", "900", "-1", "50"],
    })


@pytest.fixture
def raw_price():
    return pd.DataFrame({
        "region_name": ["NCR", "Central Luzon", "BARMM"],
        "commodity": ["Rice", "Rice", "Rice"],
        "month": ["2022-01", "2022-01", "2022-01"],
        "farmgate_price_php_kg": ["22.5", "20", ""],
    })


@pytest.fixture
def small_tables():
    rng = np.random.default_rng(0)
    months = pd.period_range("2021-01", "2022-12", freq="M").to_timestamp()
    rows = [(r, c, m, 1000 * (1 + 0.3 * np.sin(m.month)) * rng.uniform(0.8, 1.2),
             20 * (1 + 0.2 * np.cos(m.month)) * rng.uniform(0.9, 1.1))
            for r in ("NCR", "Central Luzon", "Davao Region") for c in ("Rice", "Tomato", "Onion") for m in months]
    df = pd.DataFrame(rows, columns=["region", "crop", "month", "volume", "price"])
    df["month"] = df["month"].dt.strftime("%Y-%m")
    volume = pd.DataFrame({"Region": df.region, "Crop": df.crop, "Period": df.month,
                           "Volume_MT": df.volume.round(1).astype(str)})
    price = pd.DataFrame({"region_name": df.region, "commodity": df.crop, "month": df.month,
                          "farmgate_price_php_kg": df.price.round(2).astype(str)})
    return volume, price


def write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


class TestAcquisition:
    def test_valid_file_loads_as_text(self, tmp_path, raw_volume):
        df = acquisition.load_csv(write_csv(tmp_path, "v.csv", raw_volume.iloc[:4]), config.VOLUME_RAW_SCHEMA)
        assert len(df) == 4 and all(df[c].dtype != float for c in df)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            acquisition.load_csv(tmp_path / "nope.csv", config.VOLUME_RAW_SCHEMA)

    def test_missing_required_column_raises(self, tmp_path, raw_volume):
        path = write_csv(tmp_path, "v.csv", raw_volume.drop(columns="Volume_MT"))
        with pytest.raises(SchemaError, match="Volume_MT"):
            acquisition.load_csv(path, config.VOLUME_RAW_SCHEMA)

    def test_empty_file_raises(self, tmp_path, raw_volume):
        path = write_csv(tmp_path, "v.csv", raw_volume.iloc[0:0])
        with pytest.raises(SchemaError, match="no data rows"):
            acquisition.load_csv(path, config.VOLUME_RAW_SCHEMA)

    def test_wrong_type_column_raises(self, tmp_path, raw_volume):
        bad = raw_volume.assign(Volume_MT=["abc", "def", "ghi", "jkl", "mno"])
        with pytest.raises(SchemaError, match="expected numeric"):
            acquisition.load_csv(write_csv(tmp_path, "v.csv", bad), config.VOLUME_RAW_SCHEMA)

    def test_manifest_written_and_change_detected(self, tmp_path, caplog):
        f = tmp_path / "a.csv"
        f.write_text("x\n1\n")
        manifest = tmp_path / "m.json"
        first = acquisition.record_raw_inputs([f], manifest)
        f.write_text("x\n2\n")
        with caplog.at_level(logging.WARNING):
            second = acquisition.record_raw_inputs([f], manifest)
        assert first != second and "changed" in caplog.text

    def test_download_refuses_to_overwrite_raw(self, tmp_path):
        existing = tmp_path / "raw.csv"
        existing.write_text("keep me")
        with pytest.raises(FileExistsError):
            acquisition.download_file("http://example.invalid/x.csv", existing)
        assert existing.read_text() == "keep me"


class TestCleaning:
    def test_parse_month_mixed_formats(self):
        s = pd.Series(["2022-01", "Jan 2022", "01/2022", "2022-01-15", "garbage", None])
        out = cleaning.parse_month(s)
        assert list(out[:4]) == [pd.Timestamp("2022-01-01")] * 4
        assert out[4:].isna().all()

    def test_to_numeric_clean(self):
        out = cleaning.to_numeric_clean(pd.Series(["1,234.5", " 7 ", "x", None]))
        assert out[0] == 1234.5 and out[1] == 7 and out[2:].isna().all()

    def test_label_variants_collapse(self):
        s = pd.Series(["NCR", "ncr ", "Metro Manila", "National Capital Region"])
        assert set(cleaning.standardize_labels(s, cleaning.REGION_ALIASES)) == {"NCR"}

    def test_unknown_label_kept_and_warned(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = cleaning.standardize_labels(pd.Series(["Atlantis ", "NCR"]), cleaning.REGION_ALIASES, "t")
        assert out[0] == "Atlantis" and "not in alias map" in caplog.text

    def test_clean_volume_end_to_end(self, raw_volume):
        res = cleaning.clean_volume(raw_volume)
        d = res.data
        assert d["date"].dtype.kind == "M" and d[config.VOLUME_COL].dtype == float
        assert d.duplicated(subset=config.KEY_COLS).sum() == 0        
        assert not d["date"].isna().any()                              
        assert (d[config.VOLUME_COL].dropna() >= 0).all()               
        ncr = d[(d.region == "NCR") & (d.crop == "Rice")]
        assert len(ncr) == 1 and ncr[config.VOLUME_COL].iloc[0] == 1200.5  # duplicates collapsed
        ev = res.evidence()
        assert ev.loc["distinct_region_labels", "before"] > ev.loc["distinct_region_labels", "after"]
        assert ev.loc["value_not_directly_numeric", "after"] == 0

    def test_clean_price_nonpositive_becomes_nan(self):
        raw = pd.DataFrame({"region_name": ["NCR"], "commodity": ["Rice"], "month": ["2022-01"],
                            "farmgate_price_php_kg": ["0"]})
        assert cleaning.clean_price(raw).data[config.PRICE_COL].isna().all()

    def test_duplicate_keys_averaged(self):
        raw = pd.DataFrame({"region_name": ["NCR", "ncr"], "commodity": ["Rice", "rice"],
                            "month": ["2022-01", "2022-01"], "farmgate_price_php_kg": ["20", "22"]})
        d = cleaning.clean_price(raw).data
        assert len(d) == 1 and d[config.PRICE_COL].iloc[0] == 21


class TestDocumentedDefect:
  
    @staticmethod
    def naive_standardize(s):
        return s.str.strip().str.title()

    def test_defect_reproduced_with_original_approach(self):
        out = self.naive_standardize(pd.Series(["NCR", "BARMM", "CALABARZON"]))
        assert list(out) == ["Ncr", "Barmm", "Calabarzon"]  # acronyms mangled (the bug)

    def test_fix_preserves_acronyms(self):
        out = cleaning.standardize_labels(pd.Series(["NCR", "barmm", "Calabarzon "]), cleaning.REGION_ALIASES)
        assert list(out) == ["NCR", "BARMM", "CALABARZON"]

    def test_fix_restores_cross_table_match(self):
        vol = pd.DataFrame({"region": ["NCR"], "crop": ["Rice"], "date": [pd.Timestamp("2022-01-01")],
                            config.VOLUME_COL: [10.0]})
        price = pd.DataFrame({"region": ["ncr"], "crop": ["Rice"], "date": [pd.Timestamp("2022-01-01")],
                              config.PRICE_COL: [20.0]})
        price["region"] = cleaning.standardize_labels(price["region"], cleaning.REGION_ALIASES)
        _, rep = integration.merge_volume_price(vol, price)
        assert rep["matched"] == 1


def _tables():
    d = pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"])
    vol = pd.DataFrame({"region": ["NCR"] * 3, "crop": ["Rice"] * 3, "date": d,
                        config.VOLUME_COL: [10.0, 20.0, 30.0]})
    price = pd.DataFrame({"region": ["NCR"] * 3, "crop": ["Rice", "Rice", "Corn"],
                          "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-01-01"]),
                          config.PRICE_COL: [20.0, 21.0, 15.0]})
    return vol, price


class TestIntegration:
    def test_counts_unmatched_and_totals(self):
        merged, rep = integration.merge_volume_price(*_tables())
        assert (rep["matched"], rep["volume_only"], rep["price_only"]) == (2, 1, 1)
        assert rep["n_merged"] == 4 == rep["n_volume"] + rep["n_price"] - rep["matched"]
        assert merged[config.VOLUME_COL].sum() == 60 and merged[config.PRICE_COL].sum() == 56
        assert set(merged.match_status) == {"matched", "volume_only", "price_only"}

    def test_duplicate_key_rejected(self):
        vol, price = _tables()
        with pytest.raises(DataQualityError):
            integration.merge_volume_price(pd.concat([vol, vol.iloc[:1]]), price)

    def test_no_overlap_raises(self):
        vol, price = _tables()
        price = price.assign(crop="Garlic", date=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
        with pytest.raises(integration.IntegrationError, match="no records matched"):
            integration.merge_volume_price(vol, price)

    def test_missing_column_raises(self):
        vol, price = _tables()
        with pytest.raises(SchemaError):
            integration.merge_volume_price(vol.drop(columns="crop"), price)

    def test_analysis_ready_drops_unmatched_and_nan(self):
        vol, price = _tables()
        vol.loc[0, config.VOLUME_COL] = np.nan
        merged, _ = integration.merge_volume_price(vol, price)
        ready = integration.analysis_ready(merged)
        assert len(ready) == 1 and ready[[config.VOLUME_COL, config.PRICE_COL]].notna().all().all()


class TestNumpyTask:
    @staticmethod
    def cube():
        rng = np.random.default_rng(0)
        c = rng.normal(50, 10, size=(3, 4, 24))
        c[rng.random(c.shape) < 0.1] = np.nan
        c[0, 0, :] = 7.0           # zero-variance series
        c[1, 1, :] = np.nan        # all-NaN series
        return c

    def test_vectorised_matches_loop(self):
        assert analysis.verify_zscore(self.cube())

    def test_edge_cases(self):
        z = analysis.zscore_along(self.cube())
        assert (z[0, 0] == 0).all()          # constant -> 0, not NaN/inf
        assert np.isnan(z[1, 1]).all()       # no data -> NaN
        assert np.isfinite(z[~np.isnan(z)]).all()

    def test_zscore_properties(self):
        z = analysis.zscore_along(self.cube())[2, 3]
        assert abs(np.nanmean(z)) < 1e-12 and abs(np.nanstd(z) - 1) < 1e-12

    def test_anomalies_match_pandas_groupby(self):
        rng = np.random.default_rng(1)
        dates = pd.period_range("2020-01", periods=36, freq="M").to_timestamp()
        df = pd.DataFrame([(r, c, d, rng.normal(50, 5)) for r in "AB" for c in ("x", "y") for d in dates],
                          columns=["region", "crop", "date", config.PRICE_COL])
        df.loc[5, config.PRICE_COL] = 500  # inject a spike
        g = df.groupby(["region", "crop"])[config.PRICE_COL]
        z = (df[config.PRICE_COL] - g.transform("mean")) / g.transform(lambda s: s.std(ddof=0))
        expected = int((z.abs() > 2.5).sum())
        flagged = analysis.flag_price_anomalies(df, 2.5)
        assert len(flagged) == expected >= 1
        assert 500 in flagged[config.PRICE_COL].to_list()

    def test_cube_round_trip(self):
        vol, price = _tables()
        cube, crops, regions, months = analysis.build_price_cube(price)
        assert cube.shape == (2, 1, 2) and np.isnan(cube).sum() == 1
        assert cube[list(crops).index("Rice"), 0, 1] == 21.0


class TestAnalysis:
    def test_derived_value(self):
        df = pd.DataFrame({"region": ["A", "A"], "crop": ["x", "x"],
                           "date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                           config.VOLUME_COL: [100.0, 300.0], config.PRICE_COL: [10.0, 30.0]})
        out = analysis.add_derived(df)
        assert out["value_php_m"].iloc[0] == pytest.approx(1.0)   # 100 t * 1000 kg * 10 PHP / 1e6
        assert out["vol_ratio"].tolist() == [0.5, 1.5]

    def test_benchmark_outputs_agree(self):
        bench = analysis.benchmark_zscore(n_series_list=(20,), repeats=1)
        assert bench["matches_loop"].all() and set(bench.method) == {"python_loop", "numpy_vectorised", "pandas_groupby"}
        assert "optimisation" in analysis.benchmark_conclusion(bench, 54)


class TestValidation:
    def test_missing_summary(self):
        s = validation.missing_summary(pd.DataFrame({"a": [1, None], "b": [1, 2]}))
        assert s.loc["a", "missing"] == 1 and s.loc["a", "missing_pct"] == 50.0

    def test_check_unique_key(self):
        df = pd.DataFrame({"k": [1, 1]})
        with pytest.raises(DataQualityError):
            validation.check_unique_key(df, ["k"])


def test_pipeline_end_to_end(tmp_path, small_tables):
    vp, pp = (write_csv(tmp_path, n, d) for n, d in zip(("v.csv", "p.csv"), small_tables))
    out = run_pipeline.run(tmp_path, tmp_path / "proc", tmp_path / "fig", tmp_path / "rep",
                           volume_path=vp, price_path=pp)
    assert len(out["figures"]) >= 3 and all(p.exists() for p in out["figures"])
    assert out["merge_report"]["matched"] > 0
    assert (tmp_path / "proc" / "merged.csv").exists() and (tmp_path / "rep" / "findings.md").exists()


class TestUserSuppliedDatasets:
    """People can insert their own two files, with differently named columns."""

    def _write(self, tmp_path, tables):
        v = tables[0].rename(
            columns={"Region": "Province", "Crop": "Item", "Period": "Month", "Volume_MT": "Production (MT)"})
        p = tables[1].rename(
            columns={"region_name": "Area", "commodity": "Product", "month": "Date",
                     "farmgate_price_php_kg": "Price"})
        return write_csv(tmp_path, "my_volume.csv", v), write_csv(tmp_path, "my_price.csv", p)

    def test_synonym_columns_are_detected(self, tmp_path, small_tables):
        vp, pp = self._write(tmp_path, small_tables)
        vol, price = acquisition.load_raw_tables(tmp_path, tmp_path / "m", vp, pp)
        assert list(vol.columns) == list(config.VOLUME_RENAME)
        assert list(price.columns) == list(config.PRICE_RENAME)

    def test_explicit_mapping_overrides_detection(self, tmp_path, small_tables):
        vp, pp = self._write(tmp_path, small_tables)
        odd = pd.read_csv(pp).rename(columns={"Price": "zzz"})
        pp2 = write_csv(tmp_path, "odd_price.csv", odd)
        with pytest.raises(SchemaError):
            acquisition.load_raw_tables(tmp_path, tmp_path / "m", vp, pp2)
        _, price = acquisition.load_raw_tables(tmp_path, tmp_path / "m", vp, pp2,
                                               price_cols=acquisition.parse_column_map("value=zzz"))
        assert "farmgate_price_php_kg" in price.columns

    def test_same_file_twice_is_rejected(self, tmp_path, small_tables):
        vp, _ = self._write(tmp_path, small_tables)
        with pytest.raises(SchemaError):
            acquisition.load_raw_tables(tmp_path, tmp_path / "m", vp, vp)

    def test_bad_mapping_text_is_rejected(self):
        with pytest.raises(SchemaError):
            acquisition.parse_column_map("regionProvince")
        with pytest.raises(SchemaError):
            acquisition.parse_column_map("colour=Red")

    def test_pipeline_runs_on_user_files(self, tmp_path, small_tables):
        vp, pp = self._write(tmp_path, small_tables)
        out = run_pipeline.run(tmp_path, tmp_path / "proc", tmp_path / "fig", tmp_path / "rep",
                               volume_path=vp, price_path=pp)
        assert out["merge_report"]["matched"] > 0
        assert (tmp_path / "rep" / "findings.md").exists()

    def test_cli_without_datasets_gives_clear_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(config, "RAW_DIR", tmp_path / "empty_raw")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(SystemExit):
            run_pipeline.main(["--out-dir", str(tmp_path / "out")])
        assert "no datasets supplied" in capsys.readouterr().err

    def test_cli_runs_on_user_files(self, tmp_path, small_tables):
        vp, pp = (write_csv(tmp_path, n, d) for n, d in zip(("v.csv", "p.csv"), small_tables))
        code = run_pipeline.main(["--volume", str(vp), "--price", str(pp), "--out-dir", str(tmp_path / "out")])
        assert code == 0 and (tmp_path / "out" / "reports" / "findings.md").exists()
