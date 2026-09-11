from dataclasses import replace

import pandas as pd

from .config import BacktestConfig
from .data import align_frames, apply_mask, clip_frame, load_universe, normalize_factors
from .groups import factor_weighted_long_short, quantile_returns
from .metrics import calc_coverage, calc_ic, calc_ic_stats, calc_ret_stats
from .neutralize import (
    estimate_style_loadings,
    exposure_summary,
    load_risk_frames,
    need_neutralize,
    partial_neutralize,
)
from .returns import load_returns


class BacktestResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        summary = self.get("summary")
        if isinstance(summary, pd.DataFrame):
            return "BacktestResult(\n{}\n)".format(summary.to_string())
        return "BacktestResult(keys={})".format(sorted(self.keys()))


def run_backtest(factors, cfg=None, ret=None, universe=None, **kwargs):
    """Run IC / quantile / long-short analysis on one or more factors.

    Parameters
    ----------
    factors : str | list[str] | DataFrame | dict
        Factor name(s) loaded via data_process, or already-loaded wide frames
        (index=datetime, columns=instrument).
    cfg : BacktestConfig, optional
    ret : DataFrame, optional
        If omitted, forward returns are built from cfg.buy_price / sell_price.
    universe : str | DataFrame, optional
        Overrides cfg.universe. ``is_all`` means no extra pool filter.
    """
    cfg = cfg or BacktestConfig()
    if kwargs:
        cfg = replace(cfg, **kwargs)
    if universe is not None:
        cfg = replace(cfg, universe=universe)

    factor_map = normalize_factors(factors, cfg)
    if ret is None:
        ret = load_returns(cfg)
    else:
        ret = ret.copy()
        ret.index = pd.to_datetime(ret.index)

    pool = load_universe(cfg.universe, start=cfg.start, end=cfg.end, freq=cfg.freq)
    frames = list(factor_map.values()) + [ret]
    if pool is not None:
        frames.append(pool)
    idx, cols = align_frames(frames, start=cfg.start, end=cfg.end)
    ret = clip_frame(ret, idx, cols)
    if pool is not None:
        pool = clip_frame(pool, idx, cols).fillna(False).astype(bool)
        ret = apply_mask(ret, pool)

    ic_df = pd.DataFrame(index=idx)
    rank_ic_df = pd.DataFrame(index=idx)
    coverage_df = pd.DataFrame(index=idx)
    ls_df = pd.DataFrame(index=idx)
    ls_to_df = pd.DataFrame(index=idx)
    quantile_parts = []
    qcount_parts = []
    qto_parts = []
    exposure_parts = []
    loadings_before = {}
    loadings_after = {}
    neutralized = {}

    risk_styles, risk_industry = None, None
    do_neutralize = need_neutralize(cfg)
    if do_neutralize:
        risk_styles, risk_industry = load_risk_frames(cfg, idx, cols, pool)

    for name, raw in factor_map.items():
        factor = clip_frame(raw, idx, cols)
        if pool is not None:
            factor = apply_mask(factor, pool)
        factor = factor.where(ret.notna())

        if do_neutralize:
            factor, before = partial_neutralize(
                factor,
                risk_styles or {},
                g=cfg.neutralize_g,
                industry=risk_industry,
                min_obs=cfg.neutralize_min_obs,
                standardize=True,
            )
            after = estimate_style_loadings(factor, risk_styles or {}, min_obs=cfg.neutralize_min_obs)
            loadings_before[name] = before
            loadings_after[name] = after
            neutralized[name] = factor
            if len(before.columns):
                exposure_parts.append(exposure_summary(before, after, name))

        ic_df[name] = calc_ic(factor, ret, method="pearson", min_obs=cfg.min_obs)
        rank_ic_df[name] = calc_ic(factor, ret, method="spearman", min_obs=cfg.min_obs)
        coverage_df[name] = calc_coverage(factor, pool)

        qret, qcount, qto, _labels = quantile_returns(
            factor, ret, n_groups=cfg.n_groups, direction=cfg.direction
        )
        qret = qret.add_prefix(name + ",")
        qcount = qcount.add_prefix(name + ",")
        qto = qto.add_prefix(name + ",")
        quantile_parts.append(qret)
        qcount_parts.append(qcount)
        qto_parts.append(qto)

        ls, ls_to, _w = factor_weighted_long_short(factor, ret, direction=cfg.direction)
        ls_df[name] = ls
        ls_to_df[name] = ls_to

    result = BacktestResult()
    result["config"] = cfg
    result["ic"] = ic_df
    result["rank_ic"] = rank_ic_df
    result["ic_stats"] = calc_ic_stats(ic_df)
    result["rank_ic_stats"] = calc_ic_stats(rank_ic_df)
    result["coverage"] = coverage_df
    result["coverage_stats"] = coverage_df.describe().T
    result["long_short_ret"] = ls_df
    result["long_short_turnover"] = ls_to_df
    result["long_short_ret_stats"] = calc_ret_stats(ls_df, ls_to_df, trading_days=cfg.trading_days)
    result["quantile_ret"] = pd.concat(quantile_parts, axis=1)
    result["quantile_count"] = pd.concat(qcount_parts, axis=1)
    result["quantile_turnover"] = pd.concat(qto_parts, axis=1)
    result["quantile_ret_stats"] = calc_ret_stats(
        result["quantile_ret"], result["quantile_turnover"], trading_days=cfg.trading_days
    )
    if do_neutralize:
        result["factors_neutralized"] = neutralized
        result["style_loadings"] = loadings_before
        result["style_loadings_after"] = loadings_after
        result["style_exposure"] = (
            pd.concat(exposure_parts, ignore_index=True) if exposure_parts else pd.DataFrame()
        )
    result["summary"] = _build_summary(result, cfg)

    if cfg.plot or cfg.output_dir:
        from .plot import plot_result

        plot_result(result, output_dir=cfg.output_dir, show=cfg.plot)
    return result


def _build_summary(result, cfg):
    rows = []
    for name in result["rank_ic"].columns:
        ic = result["rank_ic_stats"].loc[name]
        ls = result["long_short_ret_stats"].loc[name]
        qcol = "{},longshort".format(name)
        q = result["quantile_ret_stats"].loc[qcol] if qcol in result["quantile_ret_stats"].index else None
        cov = result["coverage_stats"].loc[name, "mean"] if name in result["coverage_stats"].index else None
        rows.append(
            {
                "factor": name,
                "rank_ic": ic["ic_mean"],
                "rank_ir": ic["ir"],
                "rank_ic_t": ic["tstat"],
                "pos_ic_ratio": ic["pos_ic_ratio"],
                "ls_ann_ret": ls["annual_return"],
                "ls_sharpe": ls["annual_sharpe"],
                "ls_maxdd": ls["max_drawdown"],
                "ls_turnover": ls["turnover"],
                "q_ls_ann_ret": None if q is None else q["annual_return"],
                "q_ls_sharpe": None if q is None else q["annual_sharpe"],
                "coverage": cov,
                "n_groups": cfg.n_groups,
                "period": cfg.period,
                "cost": cfg.cost,
                "neutralize": ",".join(cfg.neutralize_styles or ()),
            }
        )
    return pd.DataFrame(rows).set_index("factor")
