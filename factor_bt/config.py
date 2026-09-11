from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import pandas as pd

UniverseType = Union[str, pd.DataFrame, None]


@dataclass
class BacktestConfig:
    """Daily factor backtest settings.

    Return convention matches data_process.factor_analysis:
    buy at t+open_shift, sell at t+open_shift+period.
    Default is T+1 open to T+2 open (1-day hold).
    """

    start: Optional[str] = None
    end: Optional[str] = None
    freq: str = "day"
    universe: UniverseType = "is_all"
    buy_price: str = "open"
    sell_price: str = "open"
    period: int = 1
    open_shift: int = 1
    cost: float = 0.0012
    n_groups: int = 5
    direction: int = 1
    min_obs: int = 30
    use_trade_filter: bool = True
    period_average: bool = True
    trading_days: int = 242
    plot: bool = False
    output_dir: Optional[str] = None
    # Style neutralization (factor-level). Off when neutralize_styles is empty.
    neutralize_styles: Optional[Sequence[str]] = None
    neutralize_g: Optional[Dict[str, float]] = None
    neutralize_industry: bool = False
    neutralize_log_mv: bool = False
    neutralize_min_obs: int = 50
    style_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    industry_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)


@dataclass
class DomainConfig:
    """Stock-domain / market-regime analysis. Labels are fixed at decision time t."""

    start: Optional[str] = None
    end: Optional[str] = None
    universes: Sequence[str] = ("is_500", "is_2000")
    horizons: Sequence[int] = (1, 5, 10, 20)
    primary_horizon: int = 1
    n_stock_groups: int = 3
    n_quantile: int = 5
    min_obs: int = 20
    vol_lookback: int = 20
    to_lookback: int = 20
    trend_lookback: int = 20
    to_short: int = 5
    to_long: int = 60
    hist_obs: int = 252
    min_hist_obs: int = 252
    rolling_window: int = 60
    cross_stock: str = "liquidity"
    cross_market: str = "vol"
    index_ids: Optional[Dict[str, str]] = None
    freq: str = "day"
    buy_price: str = "open"
    sell_price: str = "open"
    open_shift: int = 1
    cost: float = 0.0012
    direction: int = 1
    use_trade_filter: bool = True
    period_average: bool = False
    neutralize_styles: Optional[Sequence[str]] = None
    neutralize_g: Optional[Dict[str, float]] = None
    neutralize_industry: bool = False
    neutralize_log_mv: bool = False
    neutralize_min_obs: int = 50
    style_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    industry_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    feature_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    universe_data: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    output_dir: Optional[str] = None
    config_version: str = "domain_v1"
    load_start: Optional[str] = None

    @property
    def period(self):
        return self.primary_horizon

    @property
    def max_horizon(self):
        hs = [int(h) for h in self.horizons] + [int(self.primary_horizon)]
        return max(hs)

    @property
    def universe(self):
        return None
