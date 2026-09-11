"""Shared single-factor evaluator for stock domains and market regimes.

Always computes a daily cross-sectional statistic first, then aggregates dates.
Never pools stocks across days into one correlation.
"""

import numpy as np
import pandas as pd

from .groups import assign_groups, quantile_returns
from .labels import count_spells
from .metrics import _tstat, calc_ic


def _reset_time(df):
    df = df.copy()
    df.index.name = "decision_time"
    return df.reset_index()


def daily_rank_ic(factor, ret, mask, min_obs=20):
    return calc_ic(factor.where(mask), ret.where(mask), method="spearman", min_obs=min_obs)


def daily_quantile_ls(factor, ret, mask, n_groups=5, direction=1):
    qret, qcount, _qto, _lab = quantile_returns(
        factor.where(mask), ret.where(mask), n_groups=n_groups, direction=direction
    )
    return qret["longshort"], qcount["longshort"]


def top_group_share(factor, domain_labels, universe, n_groups=5, direction=1):
    """Share of the global top factor group that sits in each domain label."""
    scored = factor.where(universe) * (1 if direction >= 0 else -1)
    groups = assign_groups(scored, n_groups)
    top = (groups == n_groups) & universe
    top_n = top.sum(axis=1).replace(0, np.nan)
    pool_n = universe.sum(axis=1).replace(0, np.nan)
    rows = []
    for label in ("low", "mid", "high"):
        in_d = (domain_labels == label) & universe
        share = (top & in_d).sum(axis=1) / top_n
        base = in_d.sum(axis=1) / pool_n
        rows.append(
            pd.DataFrame(
                {
                    "domain_label": label,
                    "top_share": share,
                    "pool_share": base,
                    "active_share": share - base,
                    "top_n": top.sum(axis=1),
                }
            )
        )
    out = pd.concat(rows)
    return _reset_time(out)


def eval_stock_domains(factor, rets, labels, universe, horizons, n_quantile=5, min_obs=20, direction=1):
    """rets: {horizon: DataFrame}. labels: {domain_name: DataFrame of low/mid/high}."""
    daily_rows = []
    for dname, lab in labels.items():
        for dlabel in ("low", "mid", "high"):
            mask = (lab == dlabel) & universe
            n = mask.sum(axis=1)
            for h, ret in rets.items():
                ic = daily_rank_ic(factor, ret, mask, min_obs=min_obs)
                ls, ls_n = daily_quantile_ls(factor, ret, mask, n_groups=n_quantile, direction=direction)
                daily_rows.append(
                    pd.DataFrame(
                        {
                            "domain_name": dname,
                            "domain_label": dlabel,
                            "regime_name": "all",
                            "regime_label": "all",
                            "horizon": int(h),
                            "n_stocks": n,
                            "ic": ic,
                            "longshort": ls,
                            "ls_n": ls_n,
                        }
                    )
                )
    daily = pd.concat(daily_rows)
    return _reset_time(daily)


def eval_market_regimes(factor, rets, market, universe, horizons, min_obs=20):
    """Full-universe daily IC, then split dates by market state."""
    daily_rows = []
    for h, ret in rets.items():
        ic_all = daily_rank_ic(factor, ret, universe, min_obs=min_obs)
        n = (factor.notna() & ret.notna() & universe).sum(axis=1)
        for rname in ("vol", "trend", "activity"):
            for rlabel in _regime_values(rname):
                pick = market[rname] == rlabel
                daily_rows.append(
                    pd.DataFrame(
                        {
                            "domain_name": "all",
                            "domain_label": "all",
                            "regime_name": rname,
                            "regime_label": rlabel,
                            "horizon": int(h),
                            "n_stocks": n.where(pick),
                            "ic": ic_all.where(pick),
                            "longshort": np.nan,
                            "ls_n": np.nan,
                        }
                    )
                )
    daily = pd.concat(daily_rows)
    return _reset_time(daily)


def eval_cross(stock_daily, market, stock_domain="liquidity", regime_name="vol"):
    """Mean of daily within-domain IC on dates with a given market state."""
    base = stock_daily[(stock_daily["domain_name"] == stock_domain) & (stock_daily["regime_name"] == "all")].copy()
    m = market[regime_name]
    base["regime_name"] = regime_name
    base["regime_label"] = base["decision_time"].map(m)
    base = base.dropna(subset=["regime_label"])
    return base


def summarize_daily(daily, market=None):
    keys = ["universe", "factor", "domain_name", "domain_label", "regime_name", "regime_label", "horizon"]
    keys = [k for k in keys if k in daily.columns]
    g = daily.dropna(subset=["ic"]).groupby(keys, sort=False)
    out = g.agg(
        n_days=("ic", "count"),
        mean_n_stocks=("n_stocks", "mean"),
        ic_mean=("ic", "mean"),
        ic_std=("ic", "std"),
        longshort_mean=("longshort", "mean"),
    ).reset_index()
    out["ir"] = out["ic_mean"] / out["ic_std"].replace(0, np.nan)
    tstats = g["ic"].apply(_tstat).reset_index(name="tstat")
    out = out.merge(tstats, on=keys, how="left")
    if market is not None and "regime_name" in out.columns:
        spells = []
        for _, row in out.iterrows():
            if row["regime_name"] == "all":
                spells.append(np.nan)
            else:
                spells.append(count_spells(market[row["regime_name"]], row["regime_label"]))
        out["n_spells"] = spells
    return out


def yearly_ic(daily):
    if daily.empty:
        return daily
    tmp = daily.dropna(subset=["ic"]).copy()
    tmp["year"] = pd.to_datetime(tmp["decision_time"]).dt.year
    keys = ["universe", "factor", "domain_name", "domain_label", "regime_name", "regime_label", "horizon", "year"]
    keys = [k for k in keys if k in tmp.columns]
    return tmp.groupby(keys, sort=False)["ic"].mean().reset_index(name="ic_mean")


def rolling_ic(daily, window=60, min_periods=20):
    if daily.empty:
        return daily
    tmp = daily.dropna(subset=["ic"]).copy()
    tmp = tmp.sort_values("decision_time")
    keys = ["universe", "factor", "domain_name", "domain_label", "regime_name", "regime_label", "horizon"]
    keys = [k for k in keys if k in tmp.columns]

    def _roll(g):
        s = g.set_index("decision_time")["ic"].sort_index()
        mp = min(window, min_periods)
        return s.rolling(window, min_periods=mp).mean()

    parts = []
    for key, g in tmp.groupby(keys, sort=False):
        rolled = _roll(g)
        frame = rolled.rename("ic_roll").reset_index()
        if not isinstance(key, tuple):
            key = (key,)
        for name, value in zip(keys, key):
            frame[name] = value
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def domain_spread(daily, domain_name, high="high", low="low"):
    """Daily IC_high - IC_low for one stock domain, all-regime rows only."""
    sub = daily[(daily["domain_name"] == domain_name) & (daily["regime_name"] == "all")]
    if sub.empty:
        return pd.DataFrame()
    keys = ["decision_time", "universe", "factor", "horizon"]
    keys = [k for k in keys if k in sub.columns]
    hi = sub[sub["domain_label"] == high][keys + ["ic"]].rename(columns={"ic": "ic_high"})
    lo = sub[sub["domain_label"] == low][keys + ["ic"]].rename(columns={"ic": "ic_low"})
    out = hi.merge(lo, on=keys, how="inner")
    out["delta_ic"] = out["ic_high"] - out["ic_low"]
    return out


def _regime_values(name):
    if name == "trend":
        return ("down", "up")
    return ("low", "high")
