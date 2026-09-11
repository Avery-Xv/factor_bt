"""Orchestrate stock-domain and market-regime labels, then run the shared evaluator."""

import json
import os
from dataclasses import asdict, replace

import pandas as pd

from .config import DomainConfig
from .data import (
    apply_mask,
    buffered_end,
    buffered_start,
    clip_frame,
    load_field,
    load_universe,
    normalize_factors,
)
from .domain_eval import (
    domain_spread,
    eval_cross,
    eval_market_regimes,
    eval_stock_domains,
    rolling_ic,
    summarize_daily,
    top_group_share,
    yearly_ic,
)
from .engine import BacktestResult
from .labels import (
    DEFAULT_INDEX_IDS,
    assign_stock_domains,
    build_market_regimes,
    build_stock_features,
    melt_market_labels,
    melt_stock_labels,
    stock_domain_stats,
)
from .neutralize import load_risk_frames, need_neutralize, partial_neutralize
from .returns import load_horizon_returns


def run_domain_analysis(factors, cfg=None, rets=None, **kwargs):
    """Label stocks/markets at close t, evaluate 1/5/10/20d returns from t+1 open."""
    cfg = cfg or DomainConfig()
    if kwargs:
        cfg = replace(cfg, **kwargs)
    cfg = replace(cfg, load_start=buffered_start(cfg.start))

    factor_map = normalize_factors(factors, cfg)
    if rets is None:
        rets = load_horizon_returns(cfg, _horizons(cfg))
    else:
        rets = {int(h): _as_dt(df) for h, df in rets.items()}

    features = _load_stock_features(cfg)
    index_close = _load_index_close(cfg)
    index_ids = dict(DEFAULT_INDEX_IDS)
    if cfg.index_ids:
        index_ids.update(cfg.index_ids)

    daily_parts = []
    stock_label_parts = []
    market_label_parts = []
    stats_parts = []
    top_parts = []
    spread_parts = []
    stock_labels_wide = {}
    market_tables = {}

    for universe_name in cfg.universes:
        pool = _load_pool(cfg, universe_name)
        hist_frames = list(features.values()) + [pool]
        idx_hist, cols_feat = _align(hist_frames, None, cfg.end)
        fcols = None
        for fr in factor_map.values():
            fcols = fr.columns if fcols is None else fcols.intersection(fr.columns)
        cols = cols_feat.intersection(fcols)
        pool_all = clip_frame(pool, idx_hist, cols).fillna(False).astype(bool)
        feat_all = {k: apply_mask(clip_frame(v, idx_hist, cols), pool_all) for k, v in features.items()}
        labels_all = assign_stock_domains(feat_all, pool_all, n_groups=cfg.n_stock_groups)

        raw_to = _raw_turnover(cfg, idx_hist, cols, pool_all)
        idx_px = _index_series(index_close, universe_name, index_ids).reindex(idx_hist)
        market_all = build_market_regimes(
            idx_px,
            raw_to,
            pool_all,
            vol_lookback=cfg.vol_lookback,
            trend_lookback=cfg.trend_lookback,
            to_short=cfg.to_short,
            to_long=cfg.to_long,
            hist=cfg.hist_obs,
            min_obs=cfg.min_hist_obs,
        )

        start_ts = pd.Timestamp(cfg.start) if cfg.start is not None else idx_hist.min()
        end_ts = pd.Timestamp(cfg.end) if cfg.end is not None else idx_hist.max()
        factor_index = list(factor_map.values())[0].index
        idx = idx_hist.intersection(factor_index)
        idx = idx[(idx >= start_ts) & (idx <= end_ts)]
        pool = pool_all.reindex(idx)
        feat = {k: v.reindex(idx) for k, v in feat_all.items()}
        labels = {k: v.reindex(idx) for k, v in labels_all.items()}
        market = market_all.reindex(idx)

        stats, _switch = stock_domain_stats(labels, feat, pool)
        stats["universe"] = universe_name
        stats_parts.append(stats)
        stock_labels_wide[universe_name] = labels
        if cfg.output_dir:
            stock_label_parts.append(melt_stock_labels(labels, feat, pool, universe_name))
        market_tables[universe_name] = market
        market_label_parts.append(melt_market_labels(market, universe_name))

        ret_u = {h: apply_mask(clip_frame(r, idx, cols), pool) for h, r in rets.items()}
        for fname, raw in factor_map.items():
            factor = apply_mask(clip_frame(raw, idx, cols), pool)
            factor = factor.where(ret_u[cfg.primary_horizon].notna())
            if need_neutralize(cfg):
                styles, industry = load_risk_frames(cfg, idx, cols, pool)
                factor, _before = partial_neutralize(
                    factor,
                    styles or {},
                    g=cfg.neutralize_g,
                    industry=industry,
                    min_obs=cfg.neutralize_min_obs,
                    standardize=True,
                )

            stock_daily = eval_stock_domains(
                factor,
                ret_u,
                labels,
                pool,
                _horizons(cfg),
                n_quantile=cfg.n_quantile,
                min_obs=cfg.min_obs,
                direction=cfg.direction,
            )
            mkt_daily = eval_market_regimes(
                factor, ret_u, market, pool, _horizons(cfg), min_obs=cfg.min_obs
            )
            cross_daily = eval_cross(stock_daily, market, cfg.cross_stock, cfg.cross_market)
            for part in (stock_daily, mkt_daily, cross_daily):
                part["universe"] = universe_name
                part["factor"] = fname
                daily_parts.append(part)

            for dname, lab in labels.items():
                share = top_group_share(
                    factor, lab, pool, n_groups=cfg.n_quantile, direction=cfg.direction
                )
                share["universe"] = universe_name
                share["factor"] = fname
                share["domain_name"] = dname
                top_parts.append(share)
            spr = domain_spread(
                stock_daily.assign(universe=universe_name, factor=fname), cfg.cross_stock
            )
            if len(spr):
                spread_parts.append(spr)

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    result = BacktestResult()
    result["config"] = cfg
    result["daily"] = daily
    stock_daily = daily[daily["regime_name"] == "all"] if len(daily) else daily
    result["stock_summary"] = summarize_daily(stock_daily)
    mkt_daily = daily[daily["domain_name"] == "all"] if len(daily) else daily
    result["market_summary"] = _market_summary(mkt_daily, market_tables)
    cross = (
        daily[(daily["domain_name"] == cfg.cross_stock) & (daily["regime_name"] == cfg.cross_market)]
        if len(daily)
        else daily
    )
    result["cross_summary"] = summarize_daily(cross)
    result["yearly"] = yearly_ic(daily)
    result["rolling"] = rolling_ic(daily, window=cfg.rolling_window)
    result["domain_stats"] = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    result["top_share"] = pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame()
    result["domain_spread"] = pd.concat(spread_parts, ignore_index=True) if spread_parts else pd.DataFrame()
    result["stock_labels"] = stock_labels_wide
    result["market_labels"] = market_tables
    result["stock_labels_long"] = (
        pd.concat(stock_label_parts, ignore_index=True) if stock_label_parts else pd.DataFrame()
    )
    result["market_labels_long"] = (
        pd.concat(market_label_parts, ignore_index=True) if market_label_parts else pd.DataFrame()
    )
    result["summary"] = _pretty_stock_summary(result["stock_summary"], cfg)

    if cfg.output_dir:
        _save(result, cfg)
    return result


def _horizons(cfg):
    hs = [int(cfg.primary_horizon)] + [int(h) for h in cfg.horizons]
    out = []
    for h in hs:
        if h not in out:
            out.append(h)
    return out


def _as_dt(df):
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out


def _align(frames, start, end):
    from .data import align_frames

    return align_frames(frames, start=start, end=end)


def _load_pool(cfg, universe_name):
    if cfg.universe_data and universe_name in cfg.universe_data:
        mask = cfg.universe_data[universe_name].copy()
        mask.index = pd.to_datetime(mask.index)
        return mask.fillna(False).astype(bool)
    start = cfg.load_start or cfg.start
    end = buffered_end(cfg.end, cfg)
    pool = load_universe(universe_name, start=start, end=end, freq=cfg.freq)
    if pool is None:
        raise ValueError("domain analysis needs a named universe such as is_500 / is_2000")
    return pool


def _load_stock_features(cfg):
    start = cfg.load_start or cfg.start
    end = buffered_end(cfg.end, cfg)
    pre = cfg.feature_data or {}

    def _get(name):
        if name in pre:
            df = pre[name].copy()
            df.index = pd.to_datetime(df.index)
            return df
        return load_field(name, start=start, end=end, freq=cfg.freq)

    mv = _get("circ_market_cap")
    turnover = _get("total_turnover")
    close = _get("close")
    return build_stock_features(
        mv, turnover, close, vol_lookback=cfg.vol_lookback, to_lookback=cfg.to_lookback
    )


def _raw_turnover(cfg, idx, cols, pool):
    pre = cfg.feature_data or {}
    if "total_turnover" in pre:
        raw = pre["total_turnover"].copy()
        raw.index = pd.to_datetime(raw.index)
    else:
        raw = load_field(
            "total_turnover",
            start=cfg.load_start or cfg.start,
            end=buffered_end(cfg.end, cfg),
            freq=cfg.freq,
        )
    return apply_mask(clip_frame(raw, idx, cols), pool)


def _load_index_close(cfg):
    pre = cfg.feature_data or {}
    if "index_close" in pre:
        df = pre["index_close"].copy()
        df.index = pd.to_datetime(df.index)
        return df
    start = cfg.load_start or cfg.start
    end = buffered_end(cfg.end, cfg)
    return load_field("close", start=start, end=end, freq="index_day")


def _index_series(index_close, universe_name, index_ids):
    col = index_ids.get(universe_name)
    if col is None:
        raise ValueError("no index id for universe {}; pass index_ids".format(universe_name))
    if col not in index_close.columns:
        raise ValueError(
            "index {} not in index_day/close columns {}".format(col, list(index_close.columns)[:8])
        )
    return index_close[col]


def _market_summary(daily, market_tables):
    if daily is None or daily.empty:
        return pd.DataFrame()
    parts = []
    for univ, market in market_tables.items():
        sub = daily[daily["universe"] == univ] if "universe" in daily.columns else daily
        parts.append(summarize_daily(sub, market=market))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _pretty_stock_summary(summary, cfg):
    if summary is None or summary.empty:
        return summary
    h0 = int(cfg.primary_horizon)
    sub = summary[(summary["regime_name"] == "all") & (summary["horizon"] == h0)]
    if sub.empty:
        sub = summary[summary["regime_name"] == "all"]
    return sub.sort_values(["universe", "factor", "domain_name", "domain_label"])


def _save(result, cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    dump = {
        "daily": "daily_eval.parquet",
        "stock_summary": "stock_summary.csv",
        "market_summary": "market_summary.csv",
        "cross_summary": "cross_summary.csv",
        "yearly": "yearly.csv",
        "rolling": "rolling.csv",
        "domain_stats": "domain_stats.csv",
        "top_share": "top_share.csv",
        "domain_spread": "domain_spread.csv",
        "stock_labels_long": "stock_labels.parquet",
        "market_labels_long": "market_labels.parquet",
        "summary": "summary.csv",
    }
    for key, fname in dump.items():
        df = result.get(key)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        path = os.path.join(cfg.output_dir, fname)
        if fname.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    meta = asdict(
        replace(cfg, style_data=None, industry_data=None, feature_data=None, universe_data=None)
    )
    meta["horizons"] = list(_horizons(cfg))
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
