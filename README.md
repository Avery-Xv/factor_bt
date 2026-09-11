# factor_bt

轻量日频因子回测，口径对齐服务器上的 `data_process.factor_analysis`，但只用 pandas，不做拥挤度、HTML 报告和 CLI。

## 做什么

- 收益：默认 T+1 开盘买、T+2 开盘卖（`open_shift=1, period=1`），扣成本；多日持仓可按天数折成日收益
- 过滤：停牌 / ST / 次新 / 涨跌停（可关）
- 股票池：`is_all`、`is_500`、`is_2000` 等缓存池，或自己传入 bool 宽表
- 指标：Rank IC / IC、等权分层、分层多空、因子加权多空、覆盖度、换手、年化统计
- 风格中性化：截面回归去掉（或打折保留）Barra 风格 / 对数市值 / 行业，再评估
- 分域诊断：股票域（市值/流动性/波动率）与市场状态（波动/趋势/活跃度）两套标签，共用按日截面评估器
- 出图：Rank IC、分层净值、多空净值

不做：拥挤度、polars、HTML 报告、自动聚类/HMM。

## 用法

```python
from factor_bt import BacktestConfig, run_backtest

res = run_backtest(
    "momentum",
    BacktestConfig(
        start="2024-01-01",
        end="2024-06-30",
        universe="is_2000",
        n_groups=5,
        cost=0.0012,
        output_dir="./output/momentum_is_2000",
    ),
)
print(res.summary)
```

只中性化市值和残差波动（`g=0`，完全去掉），行业可选：

```python
from factor_bt import DEFAULT_NEUTRALIZE_STYLES

res = run_backtest(
    "my_alpha",
    BacktestConfig(
        start="2024-01-01",
        end="2024-06-30",
        universe="is_2000",
        neutralize_styles=DEFAULT_NEUTRALIZE_STYLES,  # size, residual_volatility
        neutralize_industry=True,   # 申万一级 dummy，始终完全去掉
        neutralize_log_mv=False,    # True 时加入 log(circ_market_cap)
        # neutralize_g={"size": 0, "residual_volatility": 0.3},  # 0=去掉, 1=保留
    ),
)
print(res.summary)
print(res.style_exposure)  # 中性化前后风格载荷
```

全套 Barra 9 风格可用 `BARRA_STYLES`。`g` 缺省为 0，不会把风格做成反向暴露。

因子也可以是宽表（index=`datetime`, columns=`instrument`），或 `{name: DataFrame}`。自己算好收益时传入 `ret=`。已有风格宽表时传入 `style_data=`，不必再读缓存。

```bash
python examples/run_momentum.py --factor momentum --universe is_2000 --start 2024-01-01 --end 2024-06-30 --neutralize
```

需要 `qlib-env`（或已安装 `data_process` 的环境）才能从 `/data/file-systems/day/` 读缓存。合成数据测试不依赖它：

```bash
python -m pytest tests -q
```

## 结果

`res.summary` 是一张总表。明细与 `factor_analysis` 类似：`rank_ic`、`ic_stats`、`quantile_ret`、`long_short_ret` 及对应 `*_stats`。

分层多空是最高组减最低组等权；`long_short_ret` 是去均值后的因子加权多空（对应 `calc_factor_weight_ret`）。`direction=1` 表示因子越大越好，不自动按样本内 IC 翻符号。

## 分域诊断

股票域 \(D_{i,t}\) 和市场状态 \(S_t\) 都在 **t 日收盘** 确定，用此后 1/5/10/20 日开盘收益评估。先每天算截面 Rank IC，再按域或按日期状态汇总，不会把多日股票混成一次相关。

```python
from factor_bt import DomainConfig, run_domain_analysis

res = run_domain_analysis(
    "momentum",
    DomainConfig(
        start="2023-01-01",
        end="2024-06-30",
        universes=("is_500", "is_2000"),  # 500 用 IN000905，2000 用 IN932000
        output_dir="./output/momentum_domain",
    ),
)
print(res.summary)          # 股票域 × 期限
print(res.market_summary)   # 含 n_spells
print(res.cross_summary)    # 流动性 × 市场波动
print(res.top_share)        # 全局高分组的域占比
```

```bash
python examples/run_domain.py --factor momentum --start 2023-01-01 --end 2024-06-30
```

第一版：股票三套独立三分位（规模=`circ_market_cap`，流动性=20日均成交额，波动率=20日收益标准差）；市场三套独立两分（指数20日波动 vs 过去252日中位数、指数20日收益正负、池成交额5日/60日比 vs 历史中位数）。交叉只做流动性 × 市场波动。阈值不用未来数据；历史不够的日期标为缺失。

