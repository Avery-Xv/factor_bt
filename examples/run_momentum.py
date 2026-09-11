"""Load a cached daily factor and run the lightweight backtest.

Example:
    python examples/run_momentum.py
    python examples/run_momentum.py --factor size --universe is_500 --start 2024-01-01 --end 2024-06-30
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from factor_bt import BacktestConfig, run_backtest
from factor_bt.styles import DEFAULT_NEUTRALIZE_STYLES


def parse_args():
    p = argparse.ArgumentParser(description="Run factor_bt on a data_process cached factor")
    p.add_argument("--factor", default="momentum")
    p.add_argument("--universe", default="is_2000")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--cost", type=float, default=0.0012)
    p.add_argument("--n-groups", type=int, default=5)
    p.add_argument("--output", default=os.path.join(ROOT, "output"))
    p.add_argument(
        "--neutralize",
        nargs="*",
        default=None,
        metavar="STYLE",
        help="style names to neutralize; omit flag to skip, pass flag with no names for size+residual_volatility",
    )
    p.add_argument("--neutralize-industry", action="store_true")
    p.add_argument("--neutralize-log-mv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.neutralize is None:
        neutralize_styles = None
        tag = args.universe
    else:
        neutralize_styles = tuple(args.neutralize) if args.neutralize else DEFAULT_NEUTRALIZE_STYLES
        tag = "{}_neut".format(args.universe)
    out_dir = os.path.join(args.output, "{}_{}".format(args.factor, tag))
    cfg = BacktestConfig(
        start=args.start,
        end=args.end,
        universe=args.universe,
        period=args.period,
        cost=args.cost,
        n_groups=args.n_groups,
        plot=False,
        output_dir=out_dir,
        neutralize_styles=neutralize_styles,
        neutralize_industry=args.neutralize_industry,
        neutralize_log_mv=args.neutralize_log_mv,
    )
    res = run_backtest(args.factor, cfg)
    print(res.summary.round(4).to_string())
    print("\nrank_ic_stats:")
    print(res.rank_ic_stats.round(4).to_string())
    if "style_exposure" in res and res.style_exposure is not None and len(res.style_exposure):
        print("\nstyle_exposure:")
        print(res.style_exposure.round(4).to_string(index=False))
    print("\nplots ->", out_dir)


if __name__ == "__main__":
    main()
