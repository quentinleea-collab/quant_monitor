# 市场底部检测器 (Market Bottom Detector)

基于 XGBoost + SHAP 的 A 股底部概率检测系统。

**核心问题**: 在持续下跌行情中，什么信号组合历史上最可靠地标记了底部？

> ⚠️ 不预测未来，只统计历史概率。

## 功能

- **底部概率评分**: 0~100 分，数值越高，历史上该信号组合触发后反弹的概率越大
- **多周期分析**: 日线 + 60 分钟 + 周线特征
- **特征贡献分析**: SHAP 值给出每个信号对底部判断的贡献度
- **高概率信号组合**: 挖掘历史上胜率最高的信号组合
- **每日扫描**: 一键输出 4 个标的的底部状态
- **输出**: 终端 + Excel + 图表

## 标的

| 代码 | 名称 | 类型 |
|------|------|------|
| 000001 | 上证指数 | 指数 |
| 399001 | 深证成指 | 指数 |
| 399006 | 创业板指 | 指数 |
| 159915 | 创业板 ETF | ETF |

## 使用

```bash
# 训练模型（首次使用必须执行）
python main.py train

# 每日扫描
python main.py scan

# 完整报告
python main.py report
```

## 输出

- 终端：底部概率表格 + 建议
- Excel: 多 Sheet 报告（扫描结果、历史统计、特征贡献、信号组合）
- 图表: 底部概率曲线、K线标注、特征贡献图、回测收益曲线

## 核心文件

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口 |
| `config.py` | 配置 |
| `feature_engineer.py` | 特征工程（30+ 信号） |
| `label_generator.py` | 训练标签生成 |
| `model_trainer.py` | XGBoost 训练 + 时间序列交叉验证 |
| `shap_analyzer.py` | SHAP 特征贡献分析 |
| `pattern_miner.py` | 高概率信号组合挖掘 |
| `daily_scanner.py` | 每日扫描引擎 |
| `reporter.py` | 报告生成 |
| `chart.py` | 图表生成 |

## 依赖

xgboost, shap, scikit-learn, pandas, numpy, matplotlib, openpyxl
