"""Cross-sectional partial style neutralization.

s = X c + u
s* = u + sum_j g_j X_j c_j

g_j=0 removes style j; g_j=1 keeps it. Industry dummies are always fully removed
when enabled. Does not reverse a style into a negative bet.
"""

import numpy as np
import pandas as pd

from .data import apply_mask, buffered_end, clip_frame, load_field, load_fields
from .styles import SW_INDUSTRIES


def need_neutralize(cfg):
    return bool(cfg.neutralize_styles) or bool(cfg.neutralize_log_mv) or bool(cfg.neutralize_industry)


def style_names_from_cfg(cfg):
    names = list(cfg.neutralize_styles or ())
    if cfg.neutralize_log_mv and "log_mv" not in names:
        names.append("log_mv")
    return names


def _clip_mask(df, idx, cols, pool):
    out = clip_frame(df, idx, cols)
    if pool is not None:
        out = apply_mask(out, pool)
    return out


def load_risk_frames(cfg, idx, cols, pool):
    """Load style / industry wide frames and align to the backtest panel."""
    start = cfg.start
    end = buffered_end(cfg.end, cfg)
    preloaded = cfg.style_data or {}

    styles = {}
    for name in cfg.neutralize_styles or ():
        if name == "log_mv":
            continue
        if name in preloaded:
            df = preloaded[name].copy()
            df.index = pd.to_datetime(df.index)
        else:
            df = load_field(name, start=start, end=end, freq=cfg.freq)
        styles[name] = _clip_mask(df, idx, cols, pool)

    if cfg.neutralize_log_mv:
        if "log_mv" in preloaded:
            df = preloaded["log_mv"].copy()
            df.index = pd.to_datetime(df.index)
        else:
            src = preloaded.get("circ_market_cap")
            if src is None:
                src = load_field("circ_market_cap", start=start, end=end, freq=cfg.freq)
            else:
                src = src.copy()
                src.index = pd.to_datetime(src.index)
            df = np.log(src.where(src > 0))
        styles["log_mv"] = _clip_mask(df, idx, cols, pool)

    industry = None
    if cfg.neutralize_industry:
        if cfg.industry_data:
            industry = {}
            for name, df in cfg.industry_data.items():
                frame = df.copy()
                frame.index = pd.to_datetime(frame.index)
                industry[name] = _clip_mask(frame, idx, cols, pool)
        else:
            loaded = load_fields(SW_INDUSTRIES, start=start, end=end, freq=cfg.freq)
            industry = {name: _clip_mask(loaded[name], idx, cols, pool) for name in SW_INDUSTRIES}
    return styles, industry


def estimate_style_loadings(factor, styles, min_obs=50):
    """Daily OLS: factor ~ 1 + styles. Returns datetime x style DataFrame."""
    if not styles:
        return pd.DataFrame(index=factor.index)
    _out, loadings = partial_neutralize(
        factor,
        styles,
        g={name: 1.0 for name in styles},
        industry=None,
        min_obs=min_obs,
        standardize=False,
        return_input_loadings=True,
    )
    return loadings


def partial_neutralize(
    factor,
    styles,
    g=None,
    industry=None,
    min_obs=50,
    standardize=True,
    return_input_loadings=True,
):
    """Neutralize a wide factor against style frames.

    Parameters
    ----------
    factor : DataFrame
        index=datetime, columns=instrument
    styles : dict[str, DataFrame]
        Style exposures, same shape convention
    g : dict[str, float], optional
        Keep-ratio per style in [0, 1]. Missing keys are 0 (fully remove).
    industry : dict[str, DataFrame], optional
        0/1 industry dummies; always fully removed when provided
    """
    if not styles and not industry:
        out = factor.copy()
        if standardize:
            out = _zscore(out)
        return out, pd.DataFrame(index=factor.index)

    style_names = list(styles.keys()) if styles else []
    g = g or {}
    g_vec = np.array([float(g.get(name, 0.0)) for name in style_names], dtype=float)
    if np.any((g_vec < 0) | (g_vec > 1)):
        raise ValueError("neutralize_g values must be in [0, 1]; do not flip style signs")

    idx, cols = factor.index, factor.columns
    y = factor.to_numpy(dtype=float)
    t_count, n_count = y.shape
    if style_names:
        style_arr = np.stack(
            [_as_numpy(styles[name], idx, cols) for name in style_names],
            axis=2,
        )
    else:
        style_arr = np.zeros((t_count, n_count, 0), dtype=float)

    ind_names = list(industry.keys()) if industry else []
    if ind_names:
        ind_arr = np.stack([_as_numpy(industry[name], idx, cols) for name in ind_names], axis=2)
    else:
        ind_arr = None

    out = np.full((t_count, n_count), np.nan, dtype=float)
    coef_style = np.full((t_count, len(style_names)), np.nan, dtype=float)
    n_style = len(style_names)

    for t in range(t_count):
        yt = y[t]
        st = style_arr[t]
        valid = np.isfinite(yt)
        if n_style:
            valid &= np.isfinite(st).all(axis=1)
        it_use = None
        if ind_arr is not None:
            it = np.nan_to_num(ind_arr[t], nan=0.0)
            keep = it.sum(axis=0) > 0
            it = it[:, keep]
            if it.shape[1] > 1:
                it = it[:, 1:]
            it_use = it
        n_valid = int(valid.sum())
        n_params = 1 + n_style + (0 if it_use is None else it_use.shape[1])
        if n_valid < max(min_obs, n_params + 5):
            continue
        pieces = [np.ones((n_valid, 1), dtype=float)]
        if n_style:
            pieces.append(st[valid])
        if it_use is not None and it_use.shape[1] > 0:
            pieces.append(it_use[valid])
        x = np.concatenate(pieces, axis=1)
        beta, *_ = np.linalg.lstsq(x, yt[valid], rcond=None)
        if n_style:
            coef_style[t] = beta[1 : 1 + n_style]
            add = st[valid] @ (g_vec * beta[1 : 1 + n_style])
        else:
            add = 0.0
        intercept = beta[0]
        style_fit = st[valid] @ beta[1 : 1 + n_style] if n_style else 0.0
        ind_fit = 0.0
        if it_use is not None and it_use.shape[1] > 0:
            ind_fit = it_use[valid] @ beta[1 + n_style :]
        resid = yt[valid] - intercept - style_fit - ind_fit
        out[t, valid] = resid + add

    result = pd.DataFrame(out, index=idx, columns=cols)
    if standardize:
        result = _zscore(result)
    loadings = pd.DataFrame(coef_style, index=idx, columns=style_names)
    if return_input_loadings:
        return result, loadings
    return result, loadings


def _as_numpy(df, idx, cols):
    return df.reindex(index=idx, columns=cols).to_numpy(dtype=float)


def _zscore(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def exposure_summary(loadings_before, loadings_after, factor_name):
    rows = []
    styles = loadings_before.columns.tolist()
    for style in styles:
        before = loadings_before[style]
        after = loadings_after[style] if style in loadings_after.columns else pd.Series(dtype=float)
        rows.append(
            {
                "factor": factor_name,
                "style": style,
                "loading_mean": before.mean(),
                "loading_abs_mean": before.abs().mean(),
                "loading_after_abs_mean": after.abs().mean() if len(after) else np.nan,
            }
        )
    return pd.DataFrame(rows)
