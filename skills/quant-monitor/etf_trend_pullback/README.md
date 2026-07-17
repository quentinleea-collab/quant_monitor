# ETF 趋势回撤分析系统 (Trend Pullback)

ETF 日 K 线趋势回撤分析：识别趋势阶段，计算盘中回撤，回测尾盘回流加仓策略。

## 功能

- 计算 MA/RSI/ATR 等核心技术指标
- 计算每日盘中回撤 (DailyPullback)
- 按趋势阶段过滤 (All / Trend / MainTrend)
- 统计分析 & 止盈线建议
- 回测尾盘回流加仓策略（胜率/持有收益/止损对比/资金曲线）
- 自动生成 Excel 报告 & PNG 图表

## 使用

```bash
# 默认配置
python main.py

# 分析单只 ETF
python main.py --code 588170

# 批量分析
python main.py --code 588170,512760,510050

# 限制日期 + 趋势模式
python main.py --start 2026-01-01 --end 2026-06-30 --mode MainTrend
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `main.py` | 入口 |
| `config.py` | 分析配置 |
| `data_loader.py` | 东方财富数据加载 |
| `indicator.py` | MA/RSI/ATR 计算 |
| `pullback.py` | 盘中回撤计算 |
| `backtest.py` | 回测引擎 |
| `statistics.py` | 统计分析 |
| `excel_export.py` | Excel 输出 |
| `chart.py` | 图表生成 |
