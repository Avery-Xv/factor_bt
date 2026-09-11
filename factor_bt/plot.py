import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_result(result, output_dir=None, show=False):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank_ic = result["rank_ic"]
    for name in rank_ic.columns:
        _plot_ic(rank_ic[name], title="{} Rank IC".format(name), path=_path(output_dir, "{}_rank_ic.png".format(name)))

    qret = result["quantile_ret"]
    for name in result["rank_ic"].columns:
        cols = [c for c in qret.columns if c.startswith(name + ",") and ",group" in c]
        if not cols:
            continue
        nav = (qret[cols].fillna(0.0) + 1.0).cumprod()
        _plot_lines(nav, title="{} quantile NAV".format(name), path=_path(output_dir, "{}_quantile.png".format(name)))

        ls_col = name + ",longshort"
        if ls_col in qret.columns:
            ls_nav = (qret[ls_col].fillna(0.0) + 1.0).cumprod()
            _plot_lines(
                ls_nav.to_frame("quantile long-short"),
                title="{} quantile long-short NAV".format(name),
                path=_path(output_dir, "{}_quantile_ls.png".format(name)),
            )

    ls = result["long_short_ret"]
    ls_nav = (ls.fillna(0.0) + 1.0).cumprod()
    _plot_lines(ls_nav, title="factor-weighted long-short NAV", path=_path(output_dir, "long_short.png"))

    if show:
        plt.show()
    plt.close("all")


def _path(output_dir, name):
    if not output_dir:
        return None
    return os.path.join(output_dir, name)


def _plot_ic(ic, title, path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#3b82f6" if x >= 0 else "#ef4444" for x in ic.fillna(0.0)]
    ax.bar(ic.index, ic.fillna(0.0), color=colors, width=2, alpha=0.6, linewidth=0)
    ax2 = ax.twinx()
    ax2.plot(ic.cumsum(), color="#111827", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Rank IC")
    ax2.set_ylabel("Cumulative")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_lines(df, title, path=None):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df)
    ax.legend(df.columns, loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.close(fig)
