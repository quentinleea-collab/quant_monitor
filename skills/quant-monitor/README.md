# Quant Monitor — 量化分析工作区

A股 ETF 量化分析工具集。

## 目录结构

| 文件夹 | 功能 | 入口 |
|--------|------|------|
| `support_backtest/` | 支撑位回测系统 | `python main.py --symbol 588170` |
| `market_analyzer/` | 市场底部检测器 | `python main.py scan` |
| `etf_trend_pullback/` | ETF 趋势回踩分析 | `python main.py --code 588170` |
| `quant_v3/` | 量化监控 v3 | — |
| `quant_monitor.py` | 量化风控监控脚本 | `python quant_monitor.py` |
| `quant_test.py` | 回测测试脚本 | `python quant_test.py` |

## 公共基础设施

| 文件 | 用途 |
|------|------|
| `config.json` | 全局配置（ETF列表、搜索参数、交易参数） |
| `support_backtest/data_loader.py` | 数据加载（东方财富 API，urllib 管道） |
| `support_backtest/indicator.py` | 技术指标（MA/BOLL/ATR/RSI/K线形态） |

## 快速开始

```bash
# 支撑位回测
cd support_backtest
python main.py --symbol 588170

# 市场底部检测
cd market_analyzer
python main.py scan
```

## 依赖

```bash
pip install pandas numpy akshare matplotlib openpyxl xgboost shap scikit-learn
```
