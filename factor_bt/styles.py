"""Risk / style names used by neutralization. Domain split is not handled here."""

# data_process.neutralize default Barra styles
BARRA_STYLES = (
    "momentum",
    "beta",
    "book_to_price",
    "earnings_yield",
    "liquidity",
    "size",
    "residual_volatility",
    "leverage",
    "growth",
)

# First-cut styles for "alpha good, size/vol drag"
DEFAULT_NEUTRALIZE_STYLES = (
    "size",
    "residual_volatility",
)

# Shenwan L1 industry dummies cached as day/{name}
SW_INDUSTRIES = (
    "交通运输",
    "传媒",
    "公用事业",
    "农林牧渔",
    "医药生物",
    "商贸零售",
    "国防军工",
    "基础化工",
    "家用电器",
    "建筑材料",
    "建筑装饰",
    "房地产",
    "有色金属",
    "机械设备",
    "汽车",
    "煤炭",
    "环保",
    "电力设备",
    "电子",
    "石油石化",
    "社会服务",
    "纺织服饰",
    "综合",
    "美容护理",
    "计算机",
    "轻工制造",
    "通信",
    "钢铁",
    "银行",
    "非银金融",
    "食品饮料",
)
