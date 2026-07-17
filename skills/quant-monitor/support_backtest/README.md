# 支撑位回测系统 (Support Backtest)

识别 ETF/股票的各种支撑位，回测历史可靠性，按反弹概率排名。

## 功能

- **8 种支撑位检测**: MA(5/10/20/30/60)、前低、箱体、趋势线、成交密集区、布林带下轨、整数关口
- **多周期 MA**: 日线、周线、月线、60分钟、120分钟
- **触及 & 反弹回测**: 检测价格触及支撑位后 N 日内的反弹成功率
- **Top 5 确认指标**: 极度缩量、长下影、快速收回、K线形态、多维度共振
- **综合评分排名**: 触及频率 × 反弹率 × 确认指标 → 综合得分
- **输出**: 终端报告 + Excel 多 Sheet + 5 张 PNG 图表

## 使用

```bash
# 默认: 588170, 全历史
python main.py

# 指定标的
python main.py --symbol 512760

# 批量分析
python main.py --symbol 588170,512760,510050

# 限制日期范围
python main.py --start 20250101 --end 20260701

# 跳过图表
python main.py --no-charts

# 查看所有选项
python main.py --help
```

## 输出

- `support_backtest_results/*.xlsx` — 5 Sheet Excel 报告
- `support_backtest_results/*.png` — 支撑位地图、排名图、热力图、饼图、雷达图

## 核心文件

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口，管道编排 |
| `config.py` | 所有可配置参数 |
| `data_loader.py` | urllib 管道（兼容 Python 3.14 代理） |
| `indicator.py` | MA/BOLL/ATR/RSI/K线形态 |
| `backtest.py` | 触及检测 + 反弹评估引擎 |
| `confirmation.py` | Top 5 确认指标计算 |
| `statistics.py` | 综合评分 + 排名 |
| `excel_export.py` | Excel 多 Sheet 输出 |
| `chart.py` | 5 种 PNG 图表 |
| `support_detectors/` | 插件式支撑位检测器 |
