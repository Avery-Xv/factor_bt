"""Stock-domain and market-regime labels at decision time t (close)."""

import numpy as np
import pandas as pd

from .groups import assign_groups


STOCK_DOMAIN_NAMES = ("size", "liquidity", "volatility")
STOCK_LABELS = {1: "low", 2: "mid", 3: "high"}
MARKET_REGIME_NAMES = ("vol", "trend", "activity")

DEFAULT_INDEX_IDS = {
    "is_500": "IN000905",
    "is_2000": "IN932000",
    "is_1000": "IN000852",
    "is_300": "IN000300",
}


def build_stock_features(mv, turnover, close, vol_lookback=20, to_lookback=20, min_periods=10):
    """Features known at close t: size, 20d mean turnover, 20d return std."""
    size = mv.astype(float)
    liquidity = turnover.astype(float).rolling(to_lookback, min_periods=min_periods).mean()
    ret = close.astype(float).pct_change()
    volatility = ret.rolling(vol_lookback, min_periods=min_periods).std()
    return {"size": size, "liquidity": liquidity, "volatility": volatility}


def assign_stock_domains(features, universe, n_groups=3):
    """Independent terciles within the universe. Returns {name: label DataFrame}."""
    labels = {}
    for dname, feat in features.items():
        masked = feat.where(universe)
        lab = assign_groups(masked, n_groups)
        lab = lab.where(universe)
        arr = lab.to_numpy()
        out = np.empty(arr.shape, dtype=object)
        out[:] = np.nan
        for key, label in STOCK_LABELS.items():
            out[arr == key] = label
        labels[dname] = pd.DataFrame(out, index=lab.index, columns=lab.columns)
    return labels


def stock_domain_stats(labels, features, universe):
    """Per day, per domain: n, feature range, switch rate."""
    rows = []
    switch = {}
    for name, lab in labels.items():
        feat = features[name]
        switch[name] = _label_switch_rate(lab, universe)
        for label in ("low", "mid", "high"):
            mask = (lab == label) & universe
            n = mask.sum(axis=1)
            vals = feat.where(mask)
            rows.append(
                pd.DataFrame(
                    {
                        "domain_name": name,
                        "domain_label": label,
                        "n_stocks": n,
                        "feature_min": vals.min(axis=1),
                        "feature_median": vals.median(axis=1),
                        "feature_max": vals.max(axis=1),
                        "switch_rate": switch[name],
                    }
                )
            )
    stats = pd.concat(rows)
    stats.index.name = "decision_time"
    return stats.reset_index(), switch


def _label_switch_rate(labels, universe):
    prev = labels.shift(1)
    both = universe & universe.shift(1).fillna(False)
    changed = (labels != prev) & labels.notna() & prev.notna()
    numer = (changed & both).sum(axis=1)
    denom = both.sum(axis=1).replace(0, np.nan)
    return (numer / denom).fillna(0.0)


def trailing_median_label(series, hist=252, min_obs=252, high="high", low="low"):
    """Compare today with median of previous `hist` observations only."""
    past = series.shift(1).rolling(hist, min_periods=min_obs).median()
    out = pd.Series(np.nan, index=series.index, dtype=object)
    valid = series.notna() & past.notna()
    out.loc[valid & (series > past)] = high
    out.loc[valid & (series <= past)] = low
    return out, past


def count_spells(labels, value):
    flag = labels == value
    if flag.empty:
        return 0
    start = flag & ~flag.shift(1, fill_value=False)
    return int(start.sum())


def build_market_regimes(
    index_close,
    turnover,
    universe,
    vol_lookback=20,
    trend_lookback=20,
    to_short=5,
    to_long=60,
    hist=252,
    min_obs=252,
    jump_n=0.05,
    jump_to=0.5,
):
    """Pool-specific market states. Labels unavailable until history is long enough."""
    px = index_close.astype(float).dropna()
    idx_ret = px.pct_change()
    vol = idx_ret.rolling(vol_lookback, min_periods=max(5, vol_lookback // 2)).std()
    trend_ret = px / px.shift(trend_lookback) - 1.0
    vol_label, vol_median = trailing_median_label(vol, hist=hist, min_obs=min_obs)

    trend_label = pd.Series(np.nan, index=px.index, dtype=object)
    ok = trend_ret.notna()
    trend_label.loc[ok & (trend_ret < 0)] = "down"
    trend_label.loc[ok & (trend_ret >= 0)] = "up"

    pool_to = turnover.where(universe).sum(axis=1)
    n_members = universe.sum(axis=1).astype(float)
    ma_short = pool_to.rolling(to_short, min_periods=max(3, to_short // 2)).mean()
    ma_long = pool_to.rolling(to_long, min_periods=max(10, to_long // 3)).mean()
    activity = ma_short / ma_long.replace(0, np.nan)
    act_label, act_median = trailing_median_label(activity, hist=hist, min_obs=min_obs)

    n_chg = n_members.diff().abs() / n_members.shift(1).replace(0, np.nan)
    to_chg = pool_to.pct_change().abs()
    rebalance_jump = (n_chg > jump_n) | (to_chg > jump_to)

    table = pd.DataFrame(
        {
            "vol": vol_label.reindex(turnover.index),
            "trend": trend_label.reindex(turnover.index),
            "activity": act_label.reindex(turnover.index),
            "vol_value": vol.reindex(turnover.index),
            "vol_threshold": vol_median.reindex(turnover.index),
            "trend_value": trend_ret.reindex(turnover.index),
            "activity_value": activity.reindex(turnover.index),
            "activity_threshold": act_median.reindex(turnover.index),
            "pool_turnover": pool_to,
            "n_members": n_members,
            "rebalance_jump": rebalance_jump.reindex(turnover.index).fillna(False),
        }
    )
    return table


def melt_stock_labels(labels, features, universe, universe_name):
    parts = []
    for name, lab in labels.items():
        stacked = lab.where(universe).stack(dropna=True)
        feat = features[name].stack(dropna=False)
        part = stacked.rename("domain_label").to_frame()
        part["feature_value"] = feat.reindex(part.index)
        part = part.reset_index()
        part.columns = ["decision_time", "stock", "domain_label", "feature_value"]
        part["universe"] = universe_name
        part["domain_name"] = name
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["decision_time", "universe", "stock", "domain_name", "domain_label", "feature_value"]
        )
    return pd.concat(parts, ignore_index=True)[
        ["decision_time", "universe", "stock", "domain_name", "domain_label", "feature_value"]
    ]


def melt_market_labels(table, universe_name):
    rows = []
    for name in MARKET_REGIME_NAMES:
        value_col = "{}_value".format(name)
        thr_col = "{}_threshold".format(name) if name != "trend" else None
        chunk = pd.DataFrame(
            {
                "decision_time": table.index,
                "universe": universe_name,
                "regime_name": name,
                "regime_label": table[name].values,
                "feature_value": table[value_col].values,
                "threshold": table[thr_col].values if thr_col in table.columns else np.nan,
                "n_members": table["n_members"].values,
                "rebalance_jump": table["rebalance_jump"].values,
            }
        )
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)
