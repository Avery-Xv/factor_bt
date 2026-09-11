"""Synthetic-data checks. Run: python -m pytest tests -q"""

import numpy as np
import pandas as pd
import pytest

from factor_bt import BacktestConfig, run_backtest
from factor_bt.groups import quantile_returns
from factor_bt.metrics import calc_ic, calc_ic_stats
from factor_bt.returns import calc_forward_return


def _make_panel(n_dates=80, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    stocks = ["S{:03d}".format(i) for i in range(n_stocks)]
    factor = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=stocks)
    noise = pd.DataFrame(rng.normal(scale=0.02, size=(n_dates, n_stocks)), index=dates, columns=stocks)
    # Daily log return at t is driven by factor[t-2], so T+1 open / T+2 open
    # (open_shift=1, period=1) lines up with factor[t].
    shock = 0.15 * factor.shift(2).fillna(0.0) + noise
    close = 10 * np.exp(shock.cumsum())
    return factor, close


def test_forward_return_horizon():
    dates = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
    ret = calc_forward_return(px, period=1, open_shift=1, cost=0.0, period_average=False)
    # signal day 0: buy day1=11, sell day2=12 -> 12/11-1
    assert ret.iloc[0, 0] == pytest.approx(12.0 / 11.0 - 1.0)


def test_rank_ic_positive_when_aligned():
    factor, close = _make_panel()
    ret = calc_forward_return(close, period=1, open_shift=1, cost=0.0)
    ic = calc_ic(factor, ret, method="spearman", min_obs=20)
    stats = calc_ic_stats(ic)
    assert stats.iloc[0]["ic_mean"] > 0.1


def test_quantile_monotonic():
    factor, close = _make_panel()
    ret = calc_forward_return(close, period=1, open_shift=1, cost=0.0)
    qret, qcount, _qto, labels = quantile_returns(factor, ret, n_groups=5, direction=1)
    means = qret[["group01", "group05"]].mean()
    assert means["group05"] > means["group01"]
    assert labels.max().max() == 5
    assert (qcount["group01"] > 0).any()


def test_run_backtest_on_frames():
    factor, close = _make_panel()
    ret = calc_forward_return(close, period=1, open_shift=1, cost=0.0)
    res = run_backtest(
        {"alpha": factor},
        BacktestConfig(universe="is_all", use_trade_filter=False, n_groups=5, cost=0.0, plot=False),
        ret=ret,
    )
    assert "alpha" in res.summary.index
    assert res.summary.loc["alpha", "rank_ic"] > 0
    assert res.quantile_ret.shape[1] >= 6
    assert res.long_short_ret["alpha"].notna().sum() > 20


def test_partial_neutralize_drops_size_keeps_alpha():
    from factor_bt.neutralize import estimate_style_loadings, partial_neutralize

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=60)
    stocks = ["S{:03d}".format(i) for i in range(80)]
    size = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    alpha = pd.DataFrame(rng.normal(size=(len(dates), len(stocks))), index=dates, columns=stocks)
    factor = 1.5 * size + alpha

    out, before = partial_neutralize(factor, {"size": size}, g={"size": 0.0}, min_obs=30)
    after = estimate_style_loadings(out, {"size": size}, min_obs=30)
    assert before["size"].abs().mean() > 0.8
    assert after["size"].abs().mean() < 0.08
    assert out.corrwith(alpha, axis=1).mean() > 0.7


def test_run_backtest_with_preloaded_styles():
    factor, close = _make_panel()
    ret = calc_forward_return(close, period=1, open_shift=1, cost=0.0)
    size = pd.DataFrame(
        np.random.default_rng(2).normal(size=factor.shape),
        index=factor.index,
        columns=factor.columns,
    )
    mixed = factor + 1.5 * size
    res = run_backtest(
        {"alpha": mixed},
        BacktestConfig(
            universe="is_all",
            use_trade_filter=False,
            n_groups=5,
            cost=0.0,
            plot=False,
            neutralize_styles=("size",),
            neutralize_min_obs=20,
            style_data={"size": size},
        ),
        ret=ret,
    )
    assert "style_exposure" in res
    row = res.style_exposure.set_index("style").loc["size"]
    assert row["loading_after_abs_mean"] < row["loading_abs_mean"]
    assert res.summary.loc["alpha", "rank_ic"] > 0
