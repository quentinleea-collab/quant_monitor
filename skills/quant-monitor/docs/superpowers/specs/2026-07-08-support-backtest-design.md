# Support Level Backtest System — Design Spec

**Date**: 2026-07-08
**Status**: approved
**Project**: `support_backtest/` (under quant-monitor workspace)

## Overview

A stock/ETF support level backtesting system that identifies multiple support types, backtests their historical reliability, and ranks support levels by probability of holding (touch frequency × rebound success rate). For each support-touch event, the top 5 key confirming indicators are computed.

## Architecture

```
support_backtest/
├── main.py                 # CLI entry point
├── config.py               # Configuration management
├── data_loader.py          # akshare data fetching (proxy-safe)
├── indicator.py            # MA/BOLL/ATR/RSI + K-line pattern detection
├── support_detectors/      # Support level detector plugins
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   ├── ma_support.py       # MA5/10/20/30/60
│   ├── prior_low.py        # Prior swing lows
│   ├── box_range.py        # Box range upper/lower bounds
│   ├── trend_line.py       # Trend line support
│   ├── volume_cluster.py   # Volume profile clusters
│   ├── bollinger.py        # Bollinger Band lower band
│   └── round_number.py     # Round number / half-point levels
├── backtest.py             # Touch detection + rebound evaluation engine
├── confirmation.py         # Top 5 confirming indicators at touch points
├── statistics.py           # Probability stats + composite scoring
├── excel_export.py         # Multi-sheet Excel report
├── chart.py                # PNG chart generation (matplotlib)
└── monitor.py              # Reserved: continuous monitoring interface
```

## Data Flow

```
akshare API (东方财富, proxy bypass)
  → data_loader.py: raw OHLCV DataFrame
    → indicator.py: append MA5~MA60, BOLL(mid/upper/lower), ATR14, RSI6,
                    volume_ma5/20, lower_shadow_ratio, candle_patterns
      → support_detectors/*: parallel detection → unified [date, type, price, extra] format
        → backtest.py: touch detection → rebound evaluation → event records
          → confirmation.py: Top 5 indicator calculation per event
            → statistics.py: aggregate scoring, rank by type & by single level
              → excel_export.py / chart.py / console output
```

## Module Details

### 1. config.py

```python
@dataclass
class SupportBacktestConfig:
    # Data
    symbol: str = "588170"           # ETF code
    start_date: str = "20000101"     # default: full history
    end_date: str = "20260708"

    # Support touch detection
    touch_tolerance: float = 0.005   # 0.5% below support = touch

    # Rebound evaluation
    rebound_periods: list = (1,2,3,5,10,20)   # days
    rebound_targets: list = (0.01, 0.02, 0.03, 0.05)  # % target

    # MA periods
    ma_periods: list = (5, 10, 20, 30, 60)

    # Prior low windows
    prior_low_windows: list = (20, 40, 60)

    # Box range lookback
    box_lookback: int = 60

    # Bollinger
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Volume cluster
    volume_profile_bins: int = 50
    volume_cluster_threshold: float = 0.7  # top 30% volume concentration

    # Round numbers
    round_intervals: list = (0.05, 0.10, 0.50, 1.00)  # for different price levels

    # Output
    output_dir: str = "support_backtest_results"
    generate_excel: bool = True
    generate_charts: bool = True
    top_n_supports: int = 30
```

### 2. data_loader.py — Proxy bypass

```python
# CRITICAL: Clear proxy before import
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

import akshare as ak

class DataLoader:
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # ak.fund_etf_hist_em() for ETFs
        # Falls back to ak.stock_zh_a_hist() for stocks
        # Returns standardized columns: date, open, high, low, close, volume, amount
```

### 3. indicator.py — Technical Indicators

Extends indicators beyond existing `etf_trend_pullback/indicator.py` with:
- MA5/10/20/30/60 (reuse existing)
- BOLL (mid=MA20, upper=mid+2σ, lower=mid-2σ)
- ATR14 (reuse existing)
- RSI6 (reuse existing)
- Volume MA5 / MA20
- Lower shadow ratio: `(min(open,close) - low) / (high - low)`
- K-line patterns: hammer, inverted_hammer, doji, bullish_engulfing, morning_star, piercing_line

### 4. support_detectors/ — Plugin Architecture

#### base.py
```python
class BaseSupportDetector(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with cols: [date, support_type, support_price]"""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    def category(self) -> str: ...
```

#### ma_support.py
For each MA period, the MA value on that day IS the support price.
- Output: `date, "MA5", price` / `date, "MA10", price` / ...

#### prior_low.py
Rolling window minimum detection:
1. Find local minima (point where low[i] < low[i-1] and low[i] < low[i+1], within N-day window)
2. The most recent prior low before each date = support price
3. Use windows: 20, 40, 60 days

#### box_range.py
Box detection algorithm:
1. Rolling 60-day window
2. Find price range [min_low, max_high]
3. If range/midpoint < 15% (consolidation), mark lower bound as support
4. Box remains active until price breaks out by >3%

#### trend_line.py
Trend line from swing lows:
1. Identify swing lows (local minima in 10-day windows)
2. Connect consecutive higher lows → ascending trend line
3. Extrapolate line to current date → support price
4. Only valid when R² > 0.7

#### volume_cluster.py
Volume profile / market profile:
1. Divide price range into N bins (50 bins over full history range)
2. Histogram volume at each price level
3. Identify top concentration zones (>=70th percentile of volume)
4. Each zone = a support level, active when price is above it

#### bollinger.py
BB lower band: `MA20 - 2 * std(20)`
Simple — each day has one BB lower band value.

#### round_number.py
Psychological levels:
1. For the ETF's price range, generate round numbers at appropriate intervals
   - Price < 1.0: intervals of 0.05
   - Price 1.0–10.0: intervals of 0.10
   - Price 10.0–50.0: intervals of 0.50
   - Price > 50.0: intervals of 1.00
2. Also include half-round numbers (0.5 intervals)
3. These are static — same levels every day

### 5. backtest.py — Touch & Rebound Engine

```python
@dataclass
class TouchEvent:
    date: str
    support_type: str          # e.g. "MA20"
    support_price: float
    actual_low: float
    touch_depth: float         # (support_price - low) / support_price, positive = pierced
    close_price: float
    # Rebound results per (period, target) combo
    rebound_results: dict      # {(period, target): (rebounded: bool, days: int, max_rebound: float)}

class BacktestEngine:
    def run(self, df: pd.DataFrame, supports_df: pd.DataFrame) -> list[TouchEvent]:
        # For each day, check each active support:
        #   1. Is low <= support * (1 + tolerance)? → TOUCH
        #   2. If touched, check rebound for each (period, target) combo
        #   3. Record TouchEvent
```

Rebound check pseudo-code:
```
for each touch_event:
    for (period, target) in rebound_combos:
        start_idx = touch_date_idx + 1
        for look_forward in 1..period:
            if close[start_idx + look_forward] >= support_price * (1 + target):
                rebound = True, days = look_forward
                break
        if not rebound:
            rebound = False, days = period
```

### 6. confirmation.py — Top 5 Confirming Indicators

Calculated at each TouchEvent date:

| # | Indicator | Formula | Threshold |
|---|-----------|---------|-----------|
| 1 | **Extreme volume shrinkage** | `volume / vol_ma5` and `volume / vol_ma20` | < 0.5 = severe shrinkage |
| 2 | **Long lower shadow** | `(min(open,close) - low) / (high - low)` | > 0.6 = long shadow |
| 3 | **Quick recovery speed** | How many days to close back above support? | Categorized: 1d/2d/3d/5d/10d/no recovery |
| 4 | **K-line pattern** | Hammer, morning star, engulfing, piercing | Detected = True/False per pattern |
| 5 | **Multi-dimension resonance** | Count how many support types converge within ±1% on same day | ≥2 = resonance, ≥3 = strong resonance |

Each TouchEvent gets a `confirmation_score` = sum of confirmed indicators (0–5).

### 7. statistics.py — Scoring & Ranking

#### Per-support-level scoring:
```
touch_count = total touches of this support
rebound_rate = successful rebounds / total touches
weighted_rebound = Σ(rebound_target_weight * rebound_success_rate)  # shorter rebound, higher weight
avg_confirmation = mean(confirmation_score at touch events)

composite_score = touch_count_weight * log(1 + touch_count)
                + rebound_weight * rebound_rate
                + confirmation_weight * avg_confirmation
```

#### Output ranking:
Two views:
1. **By support category** — grouped by type (MA/prior_low/box/bollinger/...), ranked by composite_score
2. **By individual support level** — e.g. "MA60 on 2026-03-15", all unique support instances ranked

### 8. excel_export.py — Multi-Sheet Report

Sheets:
1. **Summary** — Top 30 supports ranked, with composite score, touch count, rebound rate, top 5 indicators
2. **By Category** — One table per support category
3. **Touch Events** — Every touch event with all details
4. **Top 5 Indicators Detail** — Distribution of each confirming indicator per support type
5. **Config** — Parameters used for this run

### 9. chart.py — Visualizations

1. **Support Map Chart** — K-line with support lines overlaid
2. **Probability Ranking Bar Chart** — Horizontal bar chart of top N supports by score
3. **Rebound Rate Heatmap** — Support type × Rebound period heatmap
4. **Touch Distribution Pie** — Touch count distribution by support category
5. **Top 5 Indicator Radar Chart** — Average confirmation scores across support types

### 10. monitor.py — Reserved Interface

```python
class SupportMonitor:
    """Reserved for future continuous monitoring"""
    def check_current(self, df: pd.DataFrame) -> list[dict]:
        """Check if current price is near any support level"""
        # Returns list of active supports with distance%
    
    def should_alert(self, near_supports: list) -> bool:
        """Decision logic for alert triggering"""
```

## Proxy Issue

System has proxy `http://127.0.0.1:7890` (Clash/V2Ray) that may not be running. Solution:
- **data_loader.py** must clear proxy env vars BEFORE `import akshare`
- Works as confirmed by testing: `os.environ['HTTP_PROXY'] = ''` etc.

## Data Notes

- 588170 ETF listed ~2025-04-08, ~304 trading days of history
- akshare `fund_etf_hist_em()` supports ETFs with `adjust='qfq'` (前复权)
- Columns from akshare: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率

## Testing Strategy

- Test each detector independently with known price patterns
- Test rebound logic with synthetic data (known support, known bounce)
- Integration test: run full pipeline on 588170, verify all outputs generated
- Manual review of output Excel/charts for reasonableness

## Spec Self-Review

1. **Placeholder scan**: No TBDs or TODOs — all sections complete ✓
2. **Internal consistency**: Data flow matches architecture. Scoring matches ranking approach. All detectors fit the base interface. ✓
3. **Scope check**: 10 modules, 8 detectors, 5 indicators — self-contained, achievable in one implementation cycle. ✓
4. **Ambiguity check**: All thresholds are explicit defaults. Touch/recovery logic has pseudocode. No ambiguous terms. ✓
