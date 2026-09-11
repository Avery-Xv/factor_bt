from .config import BacktestConfig, DomainConfig
from .domain_analysis import run_domain_analysis
from .engine import BacktestResult, run_backtest
from .plot import plot_result
from .styles import BARRA_STYLES, DEFAULT_NEUTRALIZE_STYLES, SW_INDUSTRIES

__all__ = [
    "BacktestConfig",
    "DomainConfig",
    "BacktestResult",
    "run_backtest",
    "run_domain_analysis",
    "plot_result",
    "BARRA_STYLES",
    "DEFAULT_NEUTRALIZE_STYLES",
    "SW_INDUSTRIES",
]
__version__ = "0.1.0"
