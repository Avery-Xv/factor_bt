"""Run domain / regime diagnostics on a cached factor.

Example:
    python examples/run_domain.py --factor momentum --start 2023-01-01 --end 2024-06-30
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from factor_bt import DomainConfig, run_domain_analysis


def parse_args():
    p = argparse.ArgumentParser(description="Stock-domain and market-regime factor diagnostics")
    p.add_argument("--factor", default="momentum")
    p.add_argument("--universes", nargs="+", default=["is_500", "is_2000"])
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--output", default=os.path.join(ROOT, "output"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output, "{}_domain".format(args.factor))
    cfg = DomainConfig(
        start=args.start,
        end=args.end,
        universes=tuple(args.universes),
        output_dir=out_dir,
        use_trade_filter=True,
    )
    res = run_domain_analysis(args.factor, cfg)
    cols = [
        c
        for c in [
            "universe",
            "factor",
            "domain_name",
            "domain_label",
            "horizon",
            "mean_n_stocks",
            "n_days",
            "ic_mean",
            "ir",
            "tstat",
            "longshort_mean",
        ]
        if c in res.summary.columns
    ]
    print("stock domains (primary horizon)")
    print(res.summary[cols].round(4).to_string(index=False))
    print("\nmarket regimes")
    mcols = [
        c
        for c in [
            "universe",
            "regime_name",
            "regime_label",
            "horizon",
            "n_days",
            "n_spells",
            "ic_mean",
            "ir",
            "tstat",
        ]
        if c in res.market_summary.columns
    ]
    print(res.market_summary[mcols].round(4).to_string(index=False))
    print("\ncross liquidity x market vol")
    print(res.cross_summary.round(4).to_string(index=False))
    print("\nfiles ->", out_dir)


if __name__ == "__main__":
    main()
