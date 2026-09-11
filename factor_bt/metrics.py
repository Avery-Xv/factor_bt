import numpy as np
import pandas as pd


def _tstat(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(n))


def calc_ic(factor, ret, method="pearson", min_obs=30):
    n = (factor.notna() & ret.notna()).sum(axis=1)
    ic = factor.corrwith(ret, axis=1, method=method)
    return ic.where(n >= min_obs)


def calc_ic_stats(ic):
    if isinstance(ic, pd.Series):
        ic = ic.to_frame()
    rows = []
    for col in ic.columns:
        s = ic[col]
        desc = s.describe()
        mean = desc.get("mean", np.nan)
        std = desc.get("std", np.nan)
        rows.append(
            {
                "ic_mean": mean,
                "ic_std": std,
                "ir": mean / std if pd.notna(mean) and pd.notna(std) and std != 0 else np.nan,
                "tstat": _tstat(s),
                "pos_ic_ratio": (s > 0).sum() / s.notna().sum() if s.notna().any() else np.nan,
                "skew": s.skew(),
                "kurt": s.kurt(),
                "direction": np.sign(mean) if pd.notna(mean) else np.nan,
                "count": s.count(),
            }
        )
    return pd.DataFrame(rows, index=ic.columns)


def calc_ret_stats(ret, turnover=None, trading_days=242):
    if isinstance(ret, pd.Series):
        ret = ret.to_frame()
    rows = []
    for col in ret.columns:
        r = ret[col].astype(float)
        valid = r.dropna()
        n = len(valid)
        if n == 0:
            rows.append(
                {
                    "ret_mean": np.nan,
                    "ret_std": np.nan,
                    "annual_return": np.nan,
                    "annual_vol": np.nan,
                    "annual_sharpe": np.nan,
                    "winrate": np.nan,
                    "finish_return": np.nan,
                    "max_drawdown": np.nan,
                    "turnover": np.nan,
                    "fitness": np.nan,
                    "count": 0,
                }
            )
            continue
        ann_vol = valid.std(ddof=1) * np.sqrt(trading_days)
        nav = (r.fillna(0.0) + 1.0).cumprod()
        finish = float(nav.iloc[-1])
        ann_ret = finish ** (trading_days / float(n)) - 1.0
        dd = nav / nav.cummax() - 1.0
        to = np.nan
        if turnover is not None and col in getattr(turnover, "columns", []):
            to = float(turnover[col].mean())
        elif turnover is not None and isinstance(turnover, pd.Series):
            to = float(turnover.mean())
        sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan
        fitness = sharpe * abs(ann_ret) / to if pd.notna(sharpe) and pd.notna(to) and to not in (0, np.nan) else np.nan
        rows.append(
            {
                "ret_mean": float(valid.mean()),
                "ret_std": float(valid.std(ddof=1)),
                "annual_return": float(ann_ret),
                "annual_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
                "annual_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
                "winrate": float((valid > 0).mean()),
                "finish_return": finish,
                "max_drawdown": float(dd.min()),
                "turnover": to,
                "fitness": fitness,
                "count": n,
            }
        )
    return pd.DataFrame(rows, index=ret.columns)


def calc_coverage(factor, universe_mask=None):
    valid = factor.notna()
    if universe_mask is None:
        denom = factor.shape[1]
        coverage = valid.sum(axis=1) / float(denom)
    else:
        mask = universe_mask.reindex(index=factor.index, columns=factor.columns).fillna(False)
        denom = mask.sum(axis=1).replace(0, np.nan)
        coverage = (valid & mask).sum(axis=1) / denom
    return coverage
