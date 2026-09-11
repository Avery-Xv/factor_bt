import numpy as np
import pandas as pd


def assign_groups(factor, n_groups):
    pct = factor.rank(axis=1, pct=True, method="average")
    raw = np.floor(pct.to_numpy(dtype=float) * n_groups - 1e-12) + 1.0
    raw = np.clip(raw, 1, n_groups)
    raw[np.isnan(pct.to_numpy())] = np.nan
    return pd.DataFrame(raw, index=factor.index, columns=factor.columns)


def _group_mean(ret, labels, group_id):
    values = ret.to_numpy(dtype=float)
    mask = labels.to_numpy() == group_id
    mask &= np.isfinite(values)
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
    return pd.Series(mean, index=ret.index), pd.Series(count, index=ret.index, dtype=float)


def quantile_returns(factor, ret, n_groups=5, direction=1):
    scored = factor * (1 if direction >= 0 else -1)
    labels = assign_groups(scored, n_groups)
    rets = {}
    counts = {}
    for g in range(1, n_groups + 1):
        name = "group{:02d}".format(g)
        mean, count = _group_mean(ret, labels, g)
        rets[name] = mean
        counts[name] = count
    qret = pd.DataFrame(rets)
    qcount = pd.DataFrame(counts)
    top = "group{:02d}".format(n_groups)
    qret["longshort"] = qret[top] - qret["group01"]
    qcount["longshort"] = qcount[top] + qcount["group01"]
    turnover = group_turnover(labels, n_groups)
    turnover["longshort"] = turnover[top] + turnover["group01"]
    return qret, qcount, turnover, labels


def group_turnover(labels, n_groups):
    prev = labels.shift(1)
    out = {}
    for g in range(1, n_groups + 1):
        in_g = labels == g
        stay = in_g & (prev == g)
        denom = in_g.sum(axis=1).replace(0, np.nan)
        out["group{:02d}".format(g)] = (1.0 - stay.sum(axis=1) / denom).fillna(1.0)
    return pd.DataFrame(out, index=labels.index)


def factor_weighted_long_short(factor, ret, direction=1):
    """Demeaned factor-weighted long-short, same idea as data_process.calc_factor_weight_ret."""
    scored = factor * (1 if direction >= 0 else -1)
    demean = scored.sub(scored.mean(axis=1), axis=0)
    pos = demean.clip(lower=0)
    neg = demean.clip(upper=0)
    pos_sum = pos.sum(axis=1).replace(0, np.nan)
    neg_sum = neg.sum(axis=1).abs().replace(0, np.nan)
    weight = pos.div(pos_sum, axis=0).fillna(0.0) + neg.div(neg_sum, axis=0).fillna(0.0)
    ls = (weight * ret).sum(axis=1, min_count=1)
    turnover = weight.fillna(0.0).diff().abs().sum(axis=1) / weight.abs().sum(axis=1).replace(0, np.nan)
    return ls, turnover.fillna(0.0), weight
