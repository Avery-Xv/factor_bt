"""Thin loader around data_process. Callers may also pass DataFrames directly."""

from datetime import timedelta

import pandas as pd


def _as_timestamp(value, default=None):
    if value is None:
        return default
    return pd.Timestamp(value)


def load_field(name, start=None, end=None, freq="day"):
    from data_process import load_factor

    df = load_factor(name, freq=freq, limit_start=start, limit_end=end)
    if df is None:
        raise ValueError("factor '{}' not found for freq={} in {}~{}".format(name, freq, start, end))
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    return df.sort_index().sort_index(axis=1)


def load_fields(names, start=None, end=None, freq="day"):
    return {name: load_field(name, start=start, end=end, freq=freq) for name in names}


def buffered_end(end, cfg):
    if end is None:
        return None
    period = getattr(cfg, "max_horizon", None) or getattr(cfg, "period", 1)
    extra_days = int((cfg.open_shift + int(period)) * 4 + 14)
    return (_as_timestamp(end) + timedelta(days=extra_days)).strftime("%Y-%m-%d")


def buffered_start(start, extra_days=700):
    if start is None:
        return None
    return (_as_timestamp(start) - timedelta(days=int(extra_days))).strftime("%Y-%m-%d")


def load_universe(universe, start=None, end=None, freq="day"):
    if universe is None or universe == "is_all":
        return None
    if isinstance(universe, pd.DataFrame):
        mask = universe.copy()
        mask.index = pd.to_datetime(mask.index)
        return mask.fillna(False).astype(bool)
    if isinstance(universe, str):
        mask = load_field(universe, start=start, end=end, freq=freq)
        return mask.fillna(0).astype(bool)
    raise TypeError("universe must be None, 'is_all', a pool name like 'is_2000', or a boolean DataFrame")


def apply_mask(df, mask):
    if mask is None or df is None:
        return df
    mask = mask.reindex(index=df.index, columns=df.columns).fillna(False)
    return df.where(mask)


def align_frames(frames, start=None, end=None):
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("no frames to align")
    idx = frames[0].index
    cols = frames[0].columns
    for frame in frames[1:]:
        idx = idx.intersection(frame.index)
        cols = cols.intersection(frame.columns)
    start = _as_timestamp(start)
    end = _as_timestamp(end)
    if start is not None:
        idx = idx[idx >= start]
    if end is not None:
        idx = idx[idx <= end]
    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("empty intersection after aligning factor/return/universe")
    return idx, cols


def clip_frame(df, idx, cols):
    return df.reindex(index=idx, columns=cols)


def normalize_factors(factors, cfg):
    """Accept a name, list of names, DataFrame, or {name: DataFrame/name}."""
    start = cfg.start
    end = buffered_end(cfg.end, cfg)

    if isinstance(factors, pd.DataFrame):
        df = factors.copy()
        df.index = pd.to_datetime(df.index)
        return {"factor": df}

    if isinstance(factors, str):
        return {factors: load_field(factors, start=start, end=end, freq=cfg.freq)}

    if isinstance(factors, (list, tuple)):
        out = {}
        for name in factors:
            if isinstance(name, str):
                out[name] = load_field(name, start=start, end=end, freq=cfg.freq)
            else:
                raise TypeError("factor list items must be names")
        return out

    if isinstance(factors, dict):
        out = {}
        for name, value in factors.items():
            if isinstance(value, str):
                out[name] = load_field(value, start=start, end=end, freq=cfg.freq)
            elif isinstance(value, pd.DataFrame):
                df = value.copy()
                df.index = pd.to_datetime(df.index)
                out[name] = df
            else:
                raise TypeError("factor '{}' must be a name or DataFrame".format(name))
        return out

    raise TypeError("factors must be a name, list of names, DataFrame, or dict")
