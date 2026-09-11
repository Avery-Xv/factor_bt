"""Stock / market domain labels and shared evaluator."""

import numpy as np
import pandas as pd

from factor_bt import DomainConfig, run_domain_analysis
from factor_bt.labels import assign_stock_domains, build_stock_features, count_spells, trailing_median_label
from factor_bt.returns import calc_forward_return


def _panel(n_dates=150, n_stocks=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    stocks = ["S{:03d}".format(i) for i in range(n_stocks)]
    size_id = np.arange(n_stocks)
    mv = pd.DataFrame(np.tile(np.exp(size_id / 8.0), (n_dates, 1)), index=dates, columns=stocks)
    turnover = mv * (0.05 + rng.random((n_dates, n_stocks)) * 0.02)
    factor = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=stocks)
    strength = np.linspace(1.2, 0.15, n_stocks)
    noise = pd.DataFrame(rng.normal(scale=0.02, size=(n_dates, n_stocks)), index=dates, columns=stocks)
    shock = 0.12 * factor.shift(2).fillna(0.0) * strength + noise
    close = 10 * np.exp(shock.cumsum())
    idx_ret = np.concatenate(
        [rng.normal(0, 0.004, 80), rng.normal(0, 0.03, n_dates - 80)]
    )
    index_close = pd.DataFrame({"IDX": 100 * np.exp(np.cumsum(idx_ret))}, index=dates)
    universe = pd.DataFrame(True, index=dates, columns=stocks)
    return dates, stocks, factor, close, mv, turnover, index_close, universe


def test_stock_terciles_are_balanced():
    _dates, _stocks, _factor, close, mv, turnover, _idx, universe = _panel()
    feat = build_stock_features(mv, turnover, close, vol_lookback=10, to_lookback=10, min_periods=5)
    labels = assign_stock_domains(feat, universe, n_groups=3)
    row = labels["size"].iloc[-1]
    counts = row.value_counts()
    assert set(counts.index) <= {"low", "mid", "high"}
    assert counts.max() - counts.min() <= 2


def test_trailing_median_uses_past_only():
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 5.0, 5.0])
    lab, past = trailing_median_label(s, hist=3, min_obs=3)
    assert pd.isna(lab.iloc[2])
    assert lab.iloc[3] == "low"
    assert lab.iloc[4] == "high"
    assert past.iloc[4] == 1.0


def test_spell_count():
    s = pd.Series(["low", "low", "high", "high", "low", "high", "high"])
    assert count_spells(s, "high") == 2
    assert count_spells(s, "low") == 2


def test_run_domain_analysis_synthetic(tmp_path):
    _dates, _stocks, factor, close, mv, turnover, index_close, universe = _panel()
    rets = {
        1: calc_forward_return(close, period=1, open_shift=1, cost=0.0, period_average=False),
        5: calc_forward_return(close, period=5, open_shift=1, cost=0.0, period_average=False),
    }
    cfg = DomainConfig(
        start="2019-03-01",
        end="2019-07-31",
        universes=("is_500",),
        horizons=(1, 5),
        primary_horizon=1,
        min_obs=8,
        n_quantile=3,
        vol_lookback=10,
        to_lookback=10,
        trend_lookback=10,
        to_short=5,
        to_long=15,
        hist_obs=30,
        min_hist_obs=20,
        rolling_window=15,
        use_trade_filter=False,
        index_ids={"is_500": "IDX"},
        feature_data={
            "circ_market_cap": mv,
            "total_turnover": turnover,
            "close": close,
            "index_close": index_close,
        },
        universe_data={"is_500": universe},
        output_dir=str(tmp_path),
    )
    res = run_domain_analysis({"alpha": factor}, cfg, rets=rets)
    stock = res.stock_summary
    size = stock[(stock["domain_name"] == "size") & (stock["horizon"] == 1)]
    low = float(size.loc[size["domain_label"] == "low", "ic_mean"].iloc[0])
    high = float(size.loc[size["domain_label"] == "high", "ic_mean"].iloc[0])
    assert low > high
    assert not res.market_summary.empty
    assert "n_spells" in res.market_summary.columns
    cross = res.cross_summary
    assert not cross.empty
    assert set(cross["regime_label"]).issubset({"low", "high"})
    assert (tmp_path / "daily_eval.parquet").exists()
    assert (tmp_path / "config.json").exists()
    top = res.top_share
    assert {"top_share", "pool_share", "active_share"}.issubset(top.columns)
