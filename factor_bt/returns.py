"""Forward returns and tradability filters."""

import pandas as pd

from .data import buffered_end, load_field, load_fields


DEFAULT_TRADE_LIMITS = ("is_suspended", "is_st", "is_new")


def calc_forward_return(buy_price, sell_price=None, period=1, open_shift=1, cost=0.0, period_average=True):
    """r_t = sell[t+open_shift+period] / buy[t+open_shift] - 1 - cost.

    Multi-day holds are divided by period when period_average=True,
    matching data_process.calc_ret.
    """
    if sell_price is None:
        sell_price = buy_price
    close_shift = open_shift + period
    ret = sell_price.shift(-close_shift) / buy_price.shift(-open_shift) - 1.0
    ret = ret - cost
    if period_average and abs(period) > 1:
        ret = ret / float(period)
    return ret


def load_returns(cfg, period=None, period_average=None):
    start = getattr(cfg, "load_start", None) or cfg.start
    end = buffered_end(cfg.end, cfg)
    buy = load_field(cfg.buy_price, start=start, end=end, freq=cfg.freq)
    if cfg.sell_price == cfg.buy_price:
        sell = buy
    else:
        sell = load_field(cfg.sell_price, start=start, end=end, freq=cfg.freq)
        sell = sell.reindex(index=buy.index, columns=buy.columns)
    period = cfg.period if period is None else period
    if period_average is None:
        period_average = getattr(cfg, "period_average", True)
    ret = calc_forward_return(
        buy,
        sell,
        period=period,
        open_shift=cfg.open_shift,
        cost=cfg.cost,
        period_average=period_average,
    )
    if cfg.use_trade_filter:
        blocked = load_trade_block(cfg, buy)
        ret = ret.mask(blocked.shift(-cfg.open_shift).fillna(False))
    return ret


def load_horizon_returns(cfg, horizons):
    """Holding-period returns for each horizon (not converted to daily)."""
    return {int(h): load_returns(cfg, period=int(h), period_average=False) for h in horizons}


def load_trade_block(cfg, buy_price):
    start = getattr(cfg, "load_start", None) or cfg.start
    end = buffered_end(cfg.end, cfg)
    names = list(DEFAULT_TRADE_LIMITS) + ["limit_up", "limit_down"]
    fields = load_fields(names, start=start, end=end, freq=cfg.freq)
    idx, cols = buy_price.index, buy_price.columns
    blocked = None
    for name in DEFAULT_TRADE_LIMITS:
        flag = fields[name].reindex(index=idx, columns=cols).fillna(False).astype(bool)
        blocked = flag if blocked is None else (blocked | flag)
    limit_up = fields["limit_up"].reindex(index=idx, columns=cols)
    limit_down = fields["limit_down"].reindex(index=idx, columns=cols)
    hit_limit = (buy_price >= limit_up) | (buy_price <= limit_down)
    return blocked | hit_limit.fillna(False)
