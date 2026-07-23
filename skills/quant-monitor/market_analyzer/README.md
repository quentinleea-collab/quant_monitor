# 市场底部检测器 (Market Bottom Detector)

基于 XGBoost + SHAP 的 A 股底部概率检测系统。

**核心问题**: 在持续下跌行情中，什么信号组合历史上最可靠地标记了底部？

> ⚠️ 不预测未来，只统计历史概率。

## 功能

- **底部概率评分**: 0~100 分，数值越高，历史上该信号组合触发后反弹的概率越大
- **多结局预测**: 反弹>=3% / 先触+5%止盈 / 先触-3%止损 / 最终盈利概率
- **SHAP 信号贡献**: 每个信号对今日判断贡献了多少，知道"谁有效"
- **历史相似度**: 找到历史上与当前形态最相似的时间段，输出未来走势
- **交易模拟回测**: 用你的交易规则（仓位/止损/止盈）跑历史模拟
- **多周期分析**: 日线 + 60 分钟 + 周线特征

## 快速开始

```bash
# 1. 训练模型（首次使用 + 每季度更新）
python main.py train

# 2. 每日扫描
python main.py scan

# 3. 交易回测
python main.py backtest
```

## 使用

### 指定股票代码

```bash
# 训练指定标的
python main.py train --symbol 600519                    # 单只
python main.py train --symbol 600519 000858 300750     # 多只

# 扫描
python main.py scan --symbol 600519

# 回测
python main.py backtest --symbol 600519 \
    --capital 60000 --position 20 --stop_loss 3 --take_profit 5
```

### 代码前缀规则

系统自动根据代码判断上交所/深交所：

| 代码范围 | 交易所 | 示例 |
|---------|--------|------|
| 5xxxxx | 上海 (sh) | 588170 科创50ETF, 510050 上证50 |
| 6xxxxx | 上海 (sh) | 600519 茅台, 601318 平安 |
| 0xxxxx | 深圳 (sz) | 000001 上证指, 000858 五粮液 |
| 3xxxxx | 深圳 (sz) | 300750 宁德, 399006 创业板指 |
| 1xxxxx | 深圳 (sz) | 159915 创业板ETF |

### 默认标的

| 代码 | 名称 | 类型 |
|------|------|------|
| 000001 | 上证指数 | 指数 |
| 399001 | 深证成指 | 指数 |
| 399006 | 创业板指 | 指数 |
| 159915 | 创业板 ETF | ETF |

### 所有命令

```bash
python main.py train                      # 训练模型
python main.py scan                       # 每日扫描
python main.py report                     # train + scan
python main.py backtest                   # 交易模拟回测
python main.py backtest --capital 100000  # 自定义本金
python main.py backtest --position 30     # 自定义仓位%
python main.py backtest --stop_loss 5     # 自定义止损%
python main.py backtest --entry_threshold 60  # 入场阈值
```

## 输出示例

```
  创业板ETF
    反弹>=3%概率: 74%  |  先达+5%概率: 60%  |  先触-3%概率: 8%  |  最终盈利概率: 89%
    趋势: 底部构建  |  建议: 试探仓 15%
    止损位: 3.2 (结构前低:3.2, MA60:4.0)
    当前信号贡献:
      +macd_hist: +50%
      +ma60_dev: +18%
      -vol_ratio_5: -28%
    历史相似形态 (Top 3):
      2026-03-24~04-07  相似度:96%  →  10日:+12.3%
      2025-11-12~11-25  相似度:96%  →  10日:+5.5%
      2026-03-19~04-01  相似度:95%  →  10日:+16.0%
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口 |
| `ma_config.py` | 配置（默认标的列表在此修改） |
| `feature_engineer.py` | 特征工程（30+ 信号） |
| `label_generator.py` | 多结局训练标签 |
| `model_trainer.py` | XGBoost 训练 + 时序交叉验证 |
| `shap_analyzer.py` | SHAP 特征贡献分析 |
| `pattern_miner.py` | 高概率信号组合挖掘 |
| `daily_scanner.py` | 每日扫描引擎 |
| `similarity.py` | 历史相似度分析 |
| `trading_simulator.py` | 交易规则模拟回测 |

## 依赖

```bash
pip install xgboost shap scikit-learn pandas numpy matplotlib openpyxl
```
