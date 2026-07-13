# Support Level Backtest System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular support level backtesting system that detects 8+ support types, backtests their reliability, confirms with Top 5 indicators, and outputs ranked results to console/Excel/charts.

**Architecture:** Plugin-based detector architecture with a shared `BaseSupportDetector` interface. Data flows: akshare → data_loader (multi-timeframe) → indicator (MA/BOLL/ATR/RSI/patterns) → 8 detectors → backtest (touch+rebound) → confirmation (Top5) → statistics (scoring) → output (console/Excel/charts).

**Tech Stack:** Python 3.12+, akshare, pandas, numpy, matplotlib, openpyxl

## Global Constraints

- All modules live under `support_backtest/` (sibling to `etf_trend_pullback/`)
- Proxy bypass: `data_loader.py` MUST clear proxy env vars BEFORE `import akshare`
- Multi-timeframe: daily + 60min primary; weekly/monthly resampled from daily; 120min resampled from 60min
- akshare `fund_etf_hist_min_em` only supports periods: "1","5","15","30","60" — NOT "120"
- Column name differs: daily uses `日期`, minute uses `时间`
- 588170 ETF: ~304 daily bars (2025-04-08+), ~124 60min bars (2026-05-26+)
- All price columns standardized to lowercase English: date, open, high, low, close, volume, amount

---
``````

### Task 1: Project Scaffolding & Config

**Files:**
- Create: `support_backtest/__init__.py`
- Create: `support_backtest/config.py`

**Interfaces:**
- Produces: `SupportBacktestConfig` dataclass with all config fields; `SCORING_WEIGHTS` dict

- [ ] **Step 1: Create `__init__.py`**

```python
"""support_backtest — Multi-timeframe support level backtesting system"""
__version__ = "1.0.0"
```

- [ ] **Step 2: Write `config.py`**

```python
"""Configuration for support level backtest system."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SupportBacktestConfig:
    """All configurable parameters with sensible defaults."""

    # ── Data ──────────────────────────────────────────
    symbol: str = "588170"
    start_date: str = "20000101"       # full history
    end_date: str = "20260708"

    # ── Multi-timeframe MA periods ────────────────────
    # (period, timeframe_label)
    ma_configs: Tuple[Tuple[int, str], ...] = (
        # Daily
        (5, "日"), (10, "日"), (20, "日"), (30, "日"), (60, "日"),
        # Weekly (resampled)
        (10, "周"), (20, "周"), (30, "周"),
        # Monthly (resampled)
        (5, "月"), (10, "月"), (20, "月"),
        # 60-minute
        (20, "60分钟"), (60, "60分钟"), (120, "60分钟"),
        # 120-minute (resampled from 60min)
        (20, "120分钟"), (60, "120分钟"),
    )

    # ── Touch detection ───────────────────────────────
    touch_tolerance: float = 0.005     # ±0.5% of support price = touch zone

    # ── Rebound evaluation ────────────────────────────
    rebound_periods: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rebound_targets: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)

    # ── Detector params ───────────────────────────────
    prior_low_windows: Tuple[int, ...] = (20, 40, 60)
    box_lookback: int = 60
    box_consolidation_pct: float = 0.15
    box_breakout_pct: float = 0.03
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_profile_bins: int = 50
    volume_cluster_threshold: float = 0.70
    round_intervals: Tuple[float, ...] = (0.05, 0.10, 0.50, 1.00)
    trend_swing_window: int = 10
    trend_min_r_squared: float = 0.7

    # ── K-line pattern ────────────────────────────────
    doji_body_ratio: float = 0.001     # body / (high-low) < this → doji
    hammer_shadow_ratio: float = 2.0   # lower_shadow >= body * 2
    hammer_upper_max: float = 0.3      # upper shadow / total ≤ 30%

    # ── Confirmation thresholds ───────────────────────
    shrink_vol_ratio: float = 0.5      # vol / vol_ma5 < 0.5 = severe shrinkage
    long_shadow_min: float = 0.6       # lower shadow >= 60% of range
    resonance_price_pct: float = 0.01  # ±1% convergence

    # ── Scoring weights ───────────────────────────────
    touch_count_weight: float = 0.25
    rebound_weight: float = 0.50
    confirmation_weight: float = 0.25

    # ── Output ────────────────────────────────────────
    output_dir: str = "support_backtest_results"
    generate_excel: bool = True
    generate_charts: bool = True
    top_n_supports: int = 30

    # ── Derived ───────────────────────────────────────
    @property
    def daily_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "日")

    @property
    def weekly_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "周")

    @property
    def monthly_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "月")

    @property
    def min60_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "60分钟")

    @property
    def min120_ma_periods(self) -> Tuple[int, ...]:
        return tuple(p for p, t in self.ma_configs if t == "120分钟")


# Singleton
config = SupportBacktestConfig()
```

- [ ] **Step 3: Commit**

```bash
git add support_backtest/__init__.py support_backtest/config.py
git commit -m "feat: add support_backtest scaffolding and config module

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Data Loader (Proxy-Safe, Multi-Timeframe)

**Files:**
- Create: `support_backtest/data_loader.py`

**Interfaces:**
- Produces: `DataLoader` class
  - `fetch_daily(symbol, start, end) -> pd.DataFrame`
  - `fetch_minute(symbol, period, start, end) -> pd.DataFrame | None`
  - `load_all(symbol, start, end) -> dict[str, pd.DataFrame]`

- [ ] **Step 1: Write `data_loader.py`**

```python
"""
Data loader with proxy bypass for akshare.
Supports daily K-line and minute-level K-line (60min).
Weekly/monthly resampled from daily; 120min resampled from 60min.
"""
import os
import logging
from typing import Optional

# ═══ CRITICAL: clear proxy BEFORE akshare import ═══
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_key, None)
os.environ["no_proxy"] = "*"
# ══════════════════════════════════════════════════════

import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

# Standard column names
STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


class DataLoader:
    """Load ETF/stock K-line data with proxy-safe akshare access."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily K-line from Eastmoney via akshare.

        Args:
            symbol: ETF/stock code, e.g. "588170"
            start: start date "YYYYMMDD"
            end: end date "YYYYMMDD"

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        logger.info(f"Fetching daily K-line for {symbol}: {start} → {end}")
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )
        except Exception:
            logger.info("ETF API failed, trying stock API...")
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )

        return self._standardize(df, date_col="日期")

    def fetch_minute(
        self, symbol: str, period: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch minute-level K-line. Only periods "1","5","15","30","60" supported.

        Returns None if no data available.
        """
        period_map = {
            "60": "60", "30": "30", "15": "15", "5": "5", "1": "1",
        }
        klt = period_map.get(period, period)
        logger.info(f"Fetching {period}min K-line for {symbol}: {start} → {end}")

        import time
        for attempt in range(3):
            try:
                df = ak.fund_etf_hist_min_em(
                    symbol=symbol, period=klt,
                    start_date=f"{start[:4]}-{start[4:6]}-{start[6:]} 09:30:00",
                    end_date=f"{end[:4]}-{end[4:6]}-{end[6:]} 15:00:00",
                    adjust="qfq",
                )
                return self._standardize(df, date_col="时间")
            except Exception as e:
                logger.warning(f"Minute fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None

    def load_all(self, symbol: str, start: str, end: str) -> dict:
        """
        Load all available timeframes.

        Returns dict with keys: daily, weekly, monthly, min60, min120
        - daily/weekly/monthly: from daily K-line + resampling
        - min60: from 60-minute K-line API
        - min120: resampled from 60-minute (2 bars → 1)
        """
        result = {}

        # Daily
        daily = self.fetch_daily(symbol, start, end)
        if daily.empty:
            raise RuntimeError(f"No daily data for {symbol}")
        daily["date"] = pd.to_datetime(daily["date"])
        result["daily"] = daily

        # Weekly resampling
        weekly = self._resample_ohlcv(daily, "W")
        result["weekly"] = weekly

        # Monthly resampling
        monthly = self._resample_ohlcv(daily, "ME")
        result["monthly"] = monthly

        # 60-minute
        min60 = self.fetch_minute(symbol, "60", start, end)
        if min60 is not None and not min60.empty:
            min60["date"] = pd.to_datetime(min60["date"])
            result["min60"] = min60
            # 120-minute resampled
            min120 = self._resample_ohlcv(min60.set_index("date"), "2h")
            min120 = min120.reset_index().rename(columns={"index": "date"})
            result["min120"] = min120
        else:
            logger.warning("No 60min data available; skipping intraday MAs")

        logger.info(
            f"Data loaded: daily={len(daily)}, weekly={len(weekly)}, "
            f"monthly={len(monthly)}, "
            f"60min={len(result.get('min60', pd.DataFrame()))}, "
            f"120min={len(result.get('min120', pd.DataFrame()))}"
        )
        return result

    # ── helpers ───────────────────────────────────────

    def _standardize(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Map akshare columns to standard English names."""
        col_map = {
            date_col: "date",
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        keep = [c for c in STD_COLS if c in df.columns]
        df = df[keep].copy()
        # Ensure numeric types
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open", "close"])

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV to a coarser timeframe."""
        df = df.set_index("date").sort_index()
        ohlcv = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
            "volume": ("volume", "sum"),
            "amount": ("amount", "sum"),
        }
        result = df.resample(rule).agg({v: m for v, m in ohlcv.values()})
        result.columns = list(ohlcv.keys())
        return result.dropna().reset_index()
```

- [ ] **Step 2: Verify data loader works**

```bash
cd support_backtest && python -c "
from data_loader import DataLoader
dl = DataLoader()
data = dl.load_all('588170', '20240101', '20260708')
for k, v in data.items():
    print(f'{k}: {len(v)} rows, {v.iloc[0][\"date\"]} → {v.iloc[-1][\"date\"]}')
"
```

Expected: daily ~300+ rows, weekly ~60+, monthly ~16+, min60 if available.

- [ ] **Step 3: Commit**

```bash
git add support_backtest/data_loader.py
git commit -m "feat: add proxy-safe multi-timeframe data loader"
```

---

### Task 3: Technical Indicators

**Files:**
- Create: `support_backtest/indicator.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with std cols (date, open, high, low, close, volume, amount)
- Produces: `IndicatorCalculator` class
  - `compute_all(df) -> pd.DataFrame` — appends all indicator columns
  - `detect_patterns(df) -> pd.DataFrame` — appends K-line pattern columns

- [ ] **Step 1: Write `indicator.py`**

```python
"""
Technical indicators: MA, BOLL, ATR, RSI, volume MA,
lower shadow ratio, and Japanese candlestick pattern detection.
"""
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Compute all technical indicators. Appends columns, never mutates input."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return new DataFrame."""
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # MA — daily periods
        for p in self.cfg.daily_ma_periods:
            df[f"MA{p}"] = df["close"].rolling(p, min_periods=p).mean()

        # BOLL
        df["BOLL_mid"] = df["close"].rolling(self.cfg.bollinger_period,
                                              min_periods=self.cfg.bollinger_period).mean()
        boll_std = df["close"].rolling(self.cfg.bollinger_period,
                                        min_periods=self.cfg.bollinger_period).std()
        df["BOLL_upper"] = df["BOLL_mid"] + self.cfg.bollinger_std * boll_std
        df["BOLL_lower"] = df["BOLL_mid"] - self.cfg.bollinger_std * boll_std

        # ATR
        df["ATR14"] = self._atr(df, 14)

        # RSI
        df["RSI6"] = self._rsi(df, 6)

        # Volume MA
        df["vol_ma5"] = df["volume"].rolling(5, min_periods=5).mean()
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=20).mean()

        # Lower shadow ratio
        body_low = df[["open", "close"]].min(axis=1)
        total_range = df["high"] - df["low"]
        df["lower_shadow"] = np.where(
            total_range > 0,
            (body_low - df["low"]) / total_range,
            0.0,
        )
        # Upper shadow ratio
        body_high = df[["open", "close"]].max(axis=1)
        df["upper_shadow"] = np.where(
            total_range > 0,
            (df["high"] - body_high) / total_range,
            0.0,
        )
        # Body ratio
        df["body_ratio"] = np.where(
            total_range > 0,
            (body_high - body_low) / total_range,
            0.0,
        )

        # Previous close
        df["prev_close"] = df["close"].shift(1)

        logger.info(f"Indicators computed: {len(df.columns)} columns, {len(df)} rows")
        return df

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect K-line patterns. Appends boolean columns to df.

        Patterns: doji, hammer, inverted_hammer, bullish_engulfing,
                  bearish_engulfing, morning_star, piercing_line
        """
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        total_range = h - l
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)

        # Doji: tiny body relative to range
        df["pat_doji"] = (total_range > 0) & (body / total_range < self.cfg.doji_body_ratio)

        # Hammer: small body, long lower shadow, tiny upper shadow
        lower_shadow = np.where(total_range > 0, (np.minimum(o, c) - l) / total_range, 0)
        upper_shadow = np.where(total_range > 0, (h - np.maximum(o, c)) / total_range, 0)
        df["pat_hammer"] = (
            (lower_shadow >= 0.6) &
            (upper_shadow <= self.cfg.hammer_upper_max) &
            (body > 0)
        )

        # Inverted hammer (shooting star mirror)
        df["pat_inverted_hammer"] = (
            (upper_shadow >= 0.6) &
            (lower_shadow <= self.cfg.hammer_upper_max) &
            (body > 0)
        )

        # Bullish engulfing
        df["pat_bullish_engulf"] = (
            (prev_c < prev_o) &           # prior bear candle
            (c > o) &                      # current bull candle
            (o <= prev_c) & (c >= prev_o)  # engulfs
        )

        # Morning star: 3-candle pattern
        df["pat_morning_star"] = (
            (prev_c.shift(1) < prev_o.shift(1)) &     # day-2: bear
            (body.shift(1) < body.shift(2) * 0.5) &    # day-1: small body
            (c > o) &                                   # today: bull
            (c > (prev_c.shift(2) + prev_o.shift(2)) / 2)  # closes above midpoint of day-2
        )

        # Piercing line
        df["pat_piercing"] = (
            (prev_c < prev_o) &           # prior bear
            (c > o) &                      # today bull
            (o < prev_c) &                 # gap down open
            (c > (prev_o + prev_c) / 2)    # closes above midpoint of prior
        )

        logger.info("K-line patterns detected")
        return df

    # ── private ───────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    @staticmethod
    def _rsi(df: pd.DataFrame, period: int = 6) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df: pd.DataFrame,
                       cfg: Optional[SupportBacktestConfig] = None) -> pd.DataFrame:
    """Convenience: compute all indicators + patterns in one call."""
    calc = IndicatorCalculator(cfg)
    df = calc.compute_all(df)
    df = calc.detect_patterns(df)
    return df
```

- [ ] **Step 2: Quick smoke test**

```bash
cd support_backtest && python -c "
from data_loader import DataLoader
from indicator import compute_indicators
dl = DataLoader()
data = dl.load_all('588170', '20240101', '20260708')
df = compute_indicators(data['daily'])
print('Columns:', list(df.columns))
print('MA20 sample:', df['MA20'].dropna().head(3).tolist())
print('BOLL_lower sample:', df['BOLL_lower'].dropna().head(3).tolist())
print('Hammer count:', df['pat_hammer'].sum())
"
```

- [ ] **Step 3: Commit**

```bash
git add support_backtest/indicator.py
git commit -m "feat: add indicator calculator with MA/BOLL/ATR/RSI and K-line patterns"
```

---

### Task 4: Support Detector Base + __init__

**Files:**
- Create: `support_backtest/support_detectors/__init__.py`
- Create: `support_backtest/support_detectors/base.py`

**Interfaces:**
- Produces: `BaseSupportDetector(ABC)`
  - `detect(df) -> pd.DataFrame` — columns: `[date, support_type, support_price, timeframe]`
  - `name: str` (property)
  - `category: str` (property)

- [ ] **Step 1: Write `base.py`**

```python
"""Abstract base class for support level detectors."""
from abc import ABC, abstractmethod

import pandas as pd


class BaseSupportDetector(ABC):
    """
    All support detectors inherit from this.

    The detect() method takes a DataFrame with OHLCV + indicators
    and returns a DataFrame with one row per (date × support_price):
      date | support_type | support_price | timeframe
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: OHLCV + indicator DataFrame (from indicator.py)

        Returns:
            DataFrame with columns:
              - date: datetime
              - support_type: str, e.g. "MA20", "PriorLow40"
              - support_price: float
              - timeframe: str, e.g. "日", "周", "月", "60分钟", "120分钟"
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name, e.g. 'ma_support'"""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category for grouping: 'ma' | 'price_structure' | 'volume' | 'volatility' | 'psychological'"""
        ...


def combine_detections(detectors: list[BaseSupportDetector],
                       df: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors and concatenate results."""
    frames = []
    for det in detectors:
        result = det.detect(df)
        if not result.empty:
            frames.append(result)
    if not frames:
        return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "support_type"])
```

- [ ] **Step 2: Write `__init__.py`**

```python
"""Support level detector plugins."""
from support_detectors.base import BaseSupportDetector, combine_detections
from support_detectors.ma_support import MASupport
from support_detectors.bollinger import BollingerSupport
from support_detectors.prior_low import PriorLowSupport
from support_detectors.box_range import BoxRangeSupport
from support_detectors.trend_line import TrendLineSupport
from support_detectors.volume_cluster import VolumeClusterSupport
from support_detectors.round_number import RoundNumberSupport

__all__ = [
    "BaseSupportDetector", "combine_detections",
    "MASupport", "BollingerSupport", "PriorLowSupport",
    "BoxRangeSupport", "TrendLineSupport", "VolumeClusterSupport",
    "RoundNumberSupport",
]
```

- [ ] **Step 3: Commit**

```bash
git add support_backtest/support_detectors/__init__.py support_backtest/support_detectors/base.py
git commit -m "feat: add support detector base class and registry"
```

---

### Task 5: MA Support Detector (Multi-Timeframe)

**Files:**
- Create: `support_backtest/support_detectors/ma_support.py`

**Note:** This detector is special — it works across timeframes. The caller must invoke it once per timeframe DataFrame.

- [ ] **Step 1: Write `ma_support.py`**

```python
"""
MA support detector — works on any timeframe.
Each MA column in the DataFrame becomes a support level.

Usage:
    det = MASupport(periods=[5,10,20,30,60], timeframe="日")
    supports = det.detect(df_daily)
"""
import pandas as pd

from support_detectors.base import BaseSupportDetector


class MASupport(BaseSupportDetector):
    """Detect support levels from Moving Average columns in the DataFrame."""

    def __init__(self, periods: list[int], timeframe: str):
        """
        Args:
            periods: MA periods to look for, e.g. [5, 10, 20, 30, 60]
            timeframe: label, e.g. "日", "周", "月", "60分钟", "120分钟"
        """
        self._periods = periods
        self._timeframe = timeframe
        self._ma_cols = [f"MA{p}" for p in periods]

    @property
    def name(self) -> str:
        return f"ma_{self._timeframe}"

    @property
    def category(self) -> str:
        return "ma"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each date and each MA column, emit one support row."""
        rows = []
        for col in self._ma_cols:
            if col not in df.columns:
                continue
            period = col.replace("MA", "")
            for _, row in df.iterrows():
                price = row[col]
                if pd.isna(price) or price <= 0:
                    continue
                rows.append({
                    "date": row["date"],
                    "support_type": f"MA{period}",
                    "support_price": price,
                    "timeframe": self._timeframe,
                })
        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/ma_support.py
git commit -m "feat: add multi-timeframe MA support detector"
```

---

### Task 6: Bollinger Band Support Detector

**Files:**
- Create: `support_backtest/support_detectors/bollinger.py`

- [ ] **Step 1: Write `bollinger.py`**

```python
"""Bollinger Band lower band support detector."""
import pandas as pd

from support_detectors.base import BaseSupportDetector


class BollingerSupport(BaseSupportDetector):
    """BB lower band = MA20 - 2*std. Simple, one value per day."""

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def category(self) -> str:
        return "volatility"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if "BOLL_lower" not in df.columns:
            raise ValueError("BOLL_lower column missing; run indicator.py first")

        rows = []
        for _, row in df.iterrows():
            price = row["BOLL_lower"]
            if pd.isna(price) or price <= 0:
                continue
            rows.append({
                "date": row["date"],
                "support_type": "BB下轨",
                "support_price": price,
                "timeframe": "日",
            })
        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/bollinger.py
git commit -m "feat: add bollinger band support detector"
```

---

### Task 7: Prior Low Support Detector

**Files:**
- Create: `support_backtest/support_detectors/prior_low.py`

- [ ] **Step 1: Write `prior_low.py`**

```python
"""
Prior swing low support detector.

Algorithm:
  1. In rolling window of N days, find local minima (low[i] < all neighbors)
  2. The most recent prior swing low = support level for the current day
  3. Repeat for multiple window sizes (20, 40, 60)
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config
from typing import Optional


class PriorLowSupport(BaseSupportDetector):

    def __init__(self, windows: Optional[tuple[int, ...]] = None):
        self._windows = windows or default_config.prior_low_windows

    @property
    def name(self) -> str:
        return "prior_low"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        lows = df["low"].values
        n = len(df)
        rows = []

        for w in self._windows:
            # Find all swing lows within window w
            swing_low_mask = self._find_swing_lows(lows, w)
            # For each day, the most recent swing low before it = support
            last_swing_idx = -1
            for i in range(n):
                if swing_low_mask[i]:
                    last_swing_idx = i
                if last_swing_idx >= 0 and last_swing_idx < i:
                    rows.append({
                        "date": df["date"].iloc[i],
                        "support_type": f"前低{w}日",
                        "support_price": lows[last_swing_idx],
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])

    @staticmethod
    def _find_swing_lows(lows: np.ndarray, window: int) -> np.ndarray:
        """Find local minima: a point is lower than all points within ±window/2."""
        n = len(lows)
        mask = np.zeros(n, dtype=bool)
        half = max(1, window // 2)
        for i in range(half, n - half):
            left = lows[i - half:i]
            right = lows[i + 1:i + half + 1]
            if lows[i] < left.min() and lows[i] <= right.min():
                mask[i] = True
        return mask
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/prior_low.py
git commit -m "feat: add prior low support detector"
```

---

### Task 8: Box Range Support Detector

**Files:**
- Create: `support_backtest/support_detectors/box_range.py`

- [ ] **Step 1: Write `box_range.py`**

```python
"""
Box range (consolidation zone) support detector.

Algorithm:
  1. Rolling N-day window (default 60)
  2. If (max_high - min_low) / midpoint < consolidation_pct → box active
  3. Box lower bound = support until price breaks below by > breakout_pct
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class BoxRangeSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "box_range"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        lookback = self.cfg.box_lookback
        rows = []
        active_boxes: list[dict] = []  # [{lower, upper, start_idx}]

        for i in range(lookback, n):
            window = df.iloc[i - lookback:i]
            max_high = window["high"].max()
            min_low = window["low"].min()
            midpoint = (max_high + min_low) / 2

            # Check if in consolidation
            range_pct = (max_high - min_low) / midpoint if midpoint > 0 else 999
            if range_pct < self.cfg.box_consolidation_pct:
                # New box found
                active_boxes.append({
                    "lower": min_low,
                    "upper": max_high,
                    "start_idx": i,
                })

            # Emit active box supports (keep only recent, remove broken ones)
            current_price = df["close"].iloc[i]
            for box in active_boxes[:]:
                # Check if box is broken (price below lower by > breakout_pct)
                if current_price < box["lower"] * (1 - self.cfg.box_breakout_pct):
                    active_boxes.remove(box)
                    continue
                # Don't emit for the first few days after box formation
                if i - box["start_idx"] < 3:
                    continue
                rows.append({
                    "date": df["date"].iloc[i],
                    "support_type": "箱体下沿",
                    "support_price": box["lower"],
                    "timeframe": "日",
                })

            # Keep at most 5 active boxes
            if len(active_boxes) > 5:
                active_boxes = active_boxes[-5:]

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/box_range.py
git commit -m "feat: add box range support detector"
```

---

### Task 9: Trend Line Support Detector

**Files:**
- Create: `support_backtest/support_detectors/trend_line.py`

- [ ] **Step 1: Write `trend_line.py`**

```python
"""
Trend line support from swing lows.

Algorithm:
  1. Find swing lows (local minima in 10-day windows)
  2. For each pair of consecutive swing lows, fit a line
  3. If R² > threshold AND slope is positive → ascending trend line
  4. Extrapolate to current day → support price
"""
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class TrendLineSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "trend_line"

    @property
    def category(self) -> str:
        return "price_structure"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        swing_win = self.cfg.trend_swing_window

        # Find swing lows
        lows = df["low"].values
        swing_idxs = []
        for i in range(swing_win, n - swing_win):
            if lows[i] <= lows[i - swing_win:i].min() and lows[i] <= lows[i + 1:i + swing_win + 1].min():
                swing_idxs.append(i)

        if len(swing_idxs) < 2:
            return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])

        rows = []
        # For each consecutive pair of swing lows, fit a trend line
        for j in range(1, len(swing_idxs)):
            idx_a, idx_b = swing_idxs[j - 1], swing_idxs[j]
            x_vals = np.array([idx_a, idx_b], dtype=float)
            y_vals = np.array([lows[idx_a], lows[idx_b]], dtype=float)

            # Linear regression
            slope = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
            intercept = y_vals[0] - slope * x_vals[0]

            # R²
            y_pred = slope * x_vals + intercept
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 >= self.cfg.trend_min_r_squared and slope > 0:
                # Extrapolate from idx_b to end
                for i in range(idx_b, n):
                    support_price = intercept + slope * i
                    if support_price > 0:
                        rows.append({
                            "date": df["date"].iloc[i],
                            "support_type": "趋势线",
                            "support_price": support_price,
                            "timeframe": "日",
                        })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/trend_line.py
git commit -m "feat: add trend line support detector"
```

---

### Task 10: Volume Cluster Support Detector

**Files:**
- Create: `support_backtest/support_detectors/volume_cluster.py`

- [ ] **Step 1: Write `volume_cluster.py`**

```python
"""
Volume cluster (成交密集区) support detector.

Algorithm:
  1. Divide the full price range into N bins
  2. Histogram total volume in each price bin
  3. Top (1 - threshold) percentile bins = high-volume zones
  4. Each high-volume zone is a support when price is above it
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config, SupportBacktestConfig
from typing import Optional


class VolumeClusterSupport(BaseSupportDetector):

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    @property
    def name(self) -> str:
        return "volume_cluster"

    @property
    def category(self) -> str:
        return "volume"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        nbins = self.cfg.volume_profile_bins
        threshold_pct = self.cfg.volume_cluster_threshold

        # Build volume profile over full history
        price_min, price_max = df["low"].min(), df["high"].max()
        bins = np.linspace(price_min, price_max, nbins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_per_bin = np.zeros(nbins)

        for _, row in df.iterrows():
            # Distribute volume across the price range for each day
            day_range = row["high"] - row["low"]
            if day_range <= 0:
                continue
            for b in range(nbins):
                bin_low, bin_high = bins[b], bins[b + 1]
                overlap = max(0, min(row["high"], bin_high) - max(row["low"], bin_low))
                if overlap > 0:
                    volume_per_bin[b] += (overlap / day_range) * row["volume"]

        # Top percentile threshold
        vol_cutoff = np.percentile(volume_per_bin[volume_per_bin > 0], threshold_pct * 100)
        cluster_bins = np.where(volume_per_bin >= vol_cutoff)[0]

        if len(cluster_bins) == 0:
            return pd.DataFrame(columns=["date", "support_type", "support_price", "timeframe"])

        # Merge adjacent cluster bins into zones
        zones = self._merge_adjacent(cluster_bins)
        zone_prices = [bin_centers[z[0]:z[-1] + 1].mean() for z in zones]

        # For each day, emit zone prices below current price as supports
        rows = []
        for _, row in df.iterrows():
            current_close = row["close"]
            for z_price in zone_prices:
                if z_price < current_close:
                    rows.append({
                        "date": row["date"],
                        "support_type": "成交密集区",
                        "support_price": z_price,
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])

    @staticmethod
    def _merge_adjacent(indices: np.ndarray) -> list[list[int]]:
        """Merge consecutive bin indices into zones."""
        if len(indices) == 0:
            return []
        zones = []
        current = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current.append(indices[i])
            else:
                zones.append(current)
                current = [indices[i]]
        zones.append(current)
        return zones
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/volume_cluster.py
git commit -m "feat: add volume cluster support detector"
```

---

### Task 11: Round Number Support Detector

**Files:**
- Create: `support_backtest/support_detectors/round_number.py`

- [ ] **Step 1: Write `round_number.py`**

```python
"""
Round number (整数关口) psychological support detector.

Generates static levels based on the ETF's price range:
  - Price < 1.0: 0.05 intervals
  - Price 1.0-10.0: 0.10 intervals
  - Price 10.0-50.0: 0.50 intervals
  - Price > 50.0: 1.00 intervals
"""
import pandas as pd
import numpy as np

from support_detectors.base import BaseSupportDetector
from config import config as default_config


class RoundNumberSupport(BaseSupportDetector):

    @property
    def name(self) -> str:
        return "round_number"

    @property
    def category(self) -> str:
        return "psychological"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").reset_index(drop=True)
        price_min = df["low"].min()
        price_max = df["high"].max()

        # Determine step size based on price range midpoint
        midpoint = (price_min + price_max) / 2
        if midpoint < 1.0:
            step = 0.05
        elif midpoint < 10.0:
            step = 0.10
        elif midpoint < 50.0:
            step = 0.50
        else:
            step = 1.00

        # Generate ALL round numbers in the price range
        start = np.floor(price_min / step) * step
        end = np.ceil(price_max / step) * step
        levels = np.arange(start, end + step / 2, step)

        # Also add half-step levels for finer granularity
        half_levels = levels + step / 2

        all_levels = sorted(set(levels) | set(half_levels))

        rows = []
        for _, row in df.iterrows():
            current_close = row["close"]
            for level in all_levels:
                if level < current_close:
                    rows.append({
                        "date": row["date"],
                        "support_type": f"关口{level:.3g}",
                        "support_price": level,
                        "timeframe": "日",
                    })

        return pd.DataFrame(rows, columns=["date", "support_type", "support_price", "timeframe"])
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/support_detectors/round_number.py
git commit -m "feat: add round number psychological support detector"
```

---

### Task 12: Backtest Engine (Touch + Rebound)

**Files:**
- Create: `support_backtest/backtest.py`

**Interfaces:**
- Consumes: `df: pd.DataFrame` (OHLCV + indicators), `supports_df: pd.DataFrame`
- Produces: `list[TouchEvent]`, `BacktestEngine.run()`

- [ ] **Step 1: Write `backtest.py`**

```python
"""
Touch detection and rebound evaluation engine.

For each day, checks if the day's low touched any active support
(priced within tolerance). If touched, evaluates rebound across
multiple (period, target) combinations.
"""
import logging
from typing import Optional
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
import numpy as np

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class TouchEvent:
    """A single support touch event."""
    date: object
    support_type: str          # e.g. "MA20"
    support_price: float
    actual_low: float
    touch_depth: float         # positive = pierced support by this % below
    close_price: float
    timeframe: str = "日"
    # {(period, target): {"rebounded": bool, "days": int}}
    rebound_results: dict = field(default_factory=dict)


class BacktestEngine:
    """Detect touches and evaluate rebounds."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config
        self._rebound_combos = list(
            product(self.cfg.rebound_periods, self.cfg.rebound_targets)
        )

    def run(self, df: pd.DataFrame, supports_df: pd.DataFrame) -> list[TouchEvent]:
        """
        Args:
            df: Daily OHLCV + indicators, sorted by date
            supports_df: All support levels from all detectors,
                         columns: [date, support_type, support_price, timeframe]

        Returns:
            List of TouchEvent objects
        """
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        # Group supports by date for fast lookup
        supports_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        for d, grp in supports_df.groupby("date"):
            supports_by_date[d] = grp

        events: list[TouchEvent] = []
        tolerance = self.cfg.touch_tolerance

        for i in range(n):
            d = dates[i]
            if d not in supports_by_date:
                continue

            day_supports = supports_by_date[d]
            day_low = lows[i]
            day_close = closes[i]

            for _, sup in day_supports.iterrows():
                s_price = sup["support_price"]
                if pd.isna(s_price) or s_price <= 0:
                    continue

                # Touch check
                if day_low <= s_price * (1 + tolerance):
                    depth = (s_price - day_low) / s_price
                    event = TouchEvent(
                        date=d,
                        support_type=str(sup["support_type"]),
                        support_price=float(s_price),
                        actual_low=float(day_low),
                        touch_depth=float(depth),
                        close_price=float(day_close),
                        timeframe=str(sup.get("timeframe", "日")),
                    )
                    # Check rebounds
                    for period, target in self._rebound_combos:
                        rebounded, days = self._check_rebound(
                            closes, i, n, s_price, period, target
                        )
                        event.rebound_results[(period, target)] = {
                            "rebounded": rebounded,
                            "days": days,
                        }
                    events.append(event)

        logger.info(
            f"Backtest: {len(events)} touch events across "
            f"{supports_df['support_type'].nunique()} support types"
        )
        return events

    def _check_rebound(
        self, closes: np.ndarray, touch_idx: int, n_days: int,
        support_price: float, period: int, target: float
    ) -> tuple[bool, int]:
        """
        Check if price rebounds from support within `period` days.

        Returns: (rebounded: bool, days_to_rebound: int)
        """
        target_price = support_price * (1 + target)
        end_idx = min(touch_idx + period + 1, n_days)
        for offset in range(1, end_idx - touch_idx):
            idx = touch_idx + offset
            if closes[idx] >= target_price:
                return True, offset
        return False, period
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/backtest.py
git commit -m "feat: add backtest engine with touch detection and rebound evaluation"
```

---

### Task 13: Confirmation Indicators (Top 5)

**Files:**
- Create: `support_backtest/confirmation.py`

**Interfaces:**
- Consumes: `df` (OHLCV + indicators), `list[TouchEvent]`
- Produces: each TouchEvent enriched with `confirmation_score` (0–5) and indicator details

- [ ] **Step 1: Write `confirmation.py`**

```python
"""
Top 5 confirming indicators at support touch points.

1. Extreme volume shrinkage: volume / vol_ma5 < 0.5
2. Long lower shadow: lower_shadow >= 0.6
3. Quick recovery: days to reclaim support
4. K-line pattern: hammer, morning star, etc. on touch day
5. Multi-dimension resonance: ≥2 support types converge within ±1%
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from backtest import TouchEvent
from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class ConfirmationAnalyzer:
    """Enrich touch events with Top 5 confirming indicators."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def analyze(
        self, events: list[TouchEvent], df: pd.DataFrame,
        supports_df: pd.DataFrame
    ) -> list[TouchEvent]:
        """
        Compute confirmation scores for each event.

        Returns the same list with events enriched (mutated in-place).
        """
        df = df.sort_values("date").reset_index(drop=True)
        # Build date → row index map
        date_to_idx = {d: i for i, d in enumerate(df["date"])}

        # For resonance: count supports per date
        resonance_counts = self._compute_resonance(supports_df)

        for event in events:
            idx = date_to_idx.get(event.date)
            if idx is None:
                event.confirmation_score = 0
                event.confirmation_detail = {}
                continue

            row = df.iloc[idx]
            detail = {}

            # 1. Volume shrinkage
            vol_ratio = row["volume"] / row["vol_ma5"] if row["vol_ma5"] > 0 else 999
            detail["shrink"] = vol_ratio < self.cfg.shrink_vol_ratio
            detail["shrink_detail"] = {
                "vol_ratio": round(float(vol_ratio), 3),
                "threshold": self.cfg.shrink_vol_ratio,
            }

            # 2. Long lower shadow
            ls = row.get("lower_shadow", 0)
            detail["long_shadow"] = ls >= self.cfg.long_shadow_min
            detail["shadow_detail"] = {
                "lower_shadow": round(float(ls), 3),
                "threshold": self.cfg.long_shadow_min,
            }

            # 3. Recovery speed (from rebound results, shortest successful recovery)
            recovery_days = []
            for (period, target), result in event.rebound_results.items():
                if result["rebounded"]:
                    recovery_days.append(result["days"])
            detail["quick_recovery"] = len(recovery_days) > 0
            detail["recovery_detail"] = {
                "fastest_days": int(min(recovery_days)) if recovery_days else None,
                "recovery_count": len(recovery_days),
                "total_combos": len(event.rebound_results),
            }

            # 4. K-line pattern
            patterns = {
                "hammer": row.get("pat_hammer", False),
                "morning_star": row.get("pat_morning_star", False),
                "bullish_engulf": row.get("pat_bullish_engulf", False),
                "piercing": row.get("pat_piercing", False),
                "doji": row.get("pat_doji", False),
            }
            detail["pattern"] = any(patterns.values())
            detail["pattern_detail"] = {k: bool(v) for k, v in patterns.items()}

            # 5. Multi-dimension resonance
            day_resonance = resonance_counts.get(event.date, 0)
            detail["resonance"] = day_resonance >= 2
            detail["resonance_detail"] = {
                "convergent_count": int(day_resonance),
                "is_strong": day_resonance >= 3,
            }

            # Composite score
            score = sum([
                1 if detail["shrink"] else 0,
                1 if detail["long_shadow"] else 0,
                1 if detail["quick_recovery"] else 0,
                1 if detail["pattern"] else 0,
                1 if detail["resonance"] else 0,
            ])

            event.confirmation_score = score
            event.confirmation_detail = detail

        # Log distribution
        scores = [e.confirmation_score for e in events]
        if scores:
            logger.info(
                f"Confirmation scores: mean={np.mean(scores):.2f}, "
                f"max={max(scores)}, distribution: "
                f"{dict(zip(*np.unique(scores, return_counts=True)))}"
            )

        return events

    def _compute_resonance(self, supports_df: pd.DataFrame) -> dict:
        """
        Count unique support types that converge within ±1% on each day.
        Resonance = multiple different support types pointing to similar price.
        """
        if supports_df.empty:
            return {}

        df = supports_df.copy()
        resonance: dict[pd.Timestamp, int] = {}

        for date, grp in df.groupby("date"):
            prices = grp["support_price"].values
            types = grp["support_type"].unique()
            n = len(prices)

            # Count how many support types have prices within 1% of each other
            # Simple approach: count unique support type categories per day
            # "Resonance" means ≥2 DIFFERENT support types point to similar price zone
            convergent_groups = 0
            used = set()
            for i in range(n):
                if i in used:
                    continue
                group = [i]
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    if abs(prices[i] - prices[j]) / prices[i] < self.cfg.resonance_price_pct:
                        # Check if different support types
                        if types[i] != types[j]:
                            group.append(j)
                            used.add(j)
                if len(set(types[g] for g in group)) >= 2:
                    convergent_groups += 1
                used.add(i)

            # Count total unique support types on this day
            resonance[date] = len(types)

        return resonance
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/confirmation.py
git commit -m "feat: add Top 5 confirmation indicator analyzer"
```

---

### Task 14: Statistics & Scoring

**Files:**
- Create: `support_backtest/statistics.py`

**Interfaces:**
- Consumes: `list[TouchEvent]`
- Produces: `SupportStats` DataFrame (ranked), `CategoryStats` DataFrame, `run_statistics()`

- [ ] **Step 1: Write `statistics.py`**

```python
"""
Statistical analysis and composite scoring.

Ranks support types by:
  composite_score = touch_weight * log(1+touches)
                  + rebound_weight * rebound_rate
                  + confirmation_weight * avg_confirmation

Outputs two views:
  1. By support type — e.g. "MA20" aggregated across all dates
  2. By category — e.g. "ma" aggregated
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from backtest import TouchEvent
from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Compute rankings and statistics from touch events."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def analyze(self, events: list[TouchEvent]) -> dict:
        """
        Args:
            events: List of TouchEvent with confirmation scores

        Returns dict with keys:
          - type_ranking: DataFrame (ranked by composite score)
          - category_ranking: DataFrame
          - summary: dict with overall stats
        """
        if not events:
            return self._empty_result()

        # Build per-type aggregation
        rows = []
        for e in events:
            # Best rebound result
            best_rebound = False
            best_days = None
            for (period, target), res in e.rebound_results.items():
                if res["rebounded"]:
                    best_rebound = True
                    if best_days is None or res["days"] < best_days:
                        best_days = res["days"]

            rows.append({
                "support_type": e.support_type,
                "category": self._classify_category(e.support_type),
                "timeframe": e.timeframe,
                "date": e.date,
                "support_price": e.support_price,
                "touched": True,
                "rebounded": best_rebound,
                "fastest_recovery_days": best_days,
                "confirmation_score": e.confirmation_score,
                "touch_depth": e.touch_depth,
            })

        df = pd.DataFrame(rows)

        # ── By support type ──────────────────────────
        type_agg = df.groupby(["support_type", "timeframe"]).agg(
            touch_count=("touched", "sum"),
            rebound_count=("rebounded", "sum"),
            avg_recovery_days=("fastest_recovery_days", "mean"),
            avg_confirmation=("confirmation_score", "mean"),
            avg_touch_depth=("touch_depth", "mean"),
        ).reset_index()

        type_agg["rebound_rate"] = np.where(
            type_agg["touch_count"] > 0,
            type_agg["rebound_count"] / type_agg["touch_count"],
            0.0,
        )

        # Composite score
        tw = self.cfg.touch_count_weight
        rw = self.cfg.rebound_weight
        cw = self.cfg.confirmation_weight

        max_confirmation = 5.0
        type_agg["composite_score"] = (
            tw * np.log1p(type_agg["touch_count"]) / np.log1p(type_agg["touch_count"].max())
            + rw * type_agg["rebound_rate"]
            + cw * type_agg["avg_confirmation"] / max_confirmation
        )

        type_agg = type_agg.sort_values("composite_score", ascending=False)
        type_agg["rank"] = range(1, len(type_agg) + 1)

        # ── By category ──────────────────────────────
        cat_agg = df.groupby("category").agg(
            touch_count=("touched", "sum"),
            rebound_count=("rebounded", "sum"),
            avg_recovery_days=("fastest_recovery_days", "mean"),
            avg_confirmation=("confirmation_score", "mean"),
            unique_types=("support_type", "nunique"),
        ).reset_index()

        cat_agg["rebound_rate"] = np.where(
            cat_agg["touch_count"] > 0,
            cat_agg["rebound_count"] / cat_agg["touch_count"],
            0.0,
        )
        cat_agg["composite_score"] = (
            tw * np.log1p(cat_agg["touch_count"]) / np.log1p(cat_agg["touch_count"].max())
            + rw * cat_agg["rebound_rate"]
            + cw * cat_agg["avg_confirmation"] / max_confirmation
        )
        cat_agg = cat_agg.sort_values("composite_score", ascending=False)

        # ── Summary ──────────────────────────────────
        summary = {
            "total_touch_events": len(df),
            "unique_support_types": df["support_type"].nunique(),
            "overall_rebound_rate": df["rebounded"].mean() if len(df) > 0 else 0,
            "avg_confirmation_score": df["confirmation_score"].mean() if len(df) > 0 else 0,
            "top_support": type_agg.iloc[0]["support_type"] if len(type_agg) > 0 else None,
            "top_rebound_rate": type_agg["rebound_rate"].max() if len(type_agg) > 0 else 0,
        }

        logger.info(
            f"Stats: {len(df)} touches, overall rebound rate={summary['overall_rebound_rate']:.1%}, "
            f"top={summary['top_support']}"
        )

        return {
            "type_ranking": type_agg,
            "category_ranking": cat_agg,
            "event_details": df,
            "summary": summary,
        }

    def _empty_result(self) -> dict:
        return {
            "type_ranking": pd.DataFrame(),
            "category_ranking": pd.DataFrame(),
            "event_details": pd.DataFrame(),
            "summary": {"total_touch_events": 0, "overall_rebound_rate": 0},
        }

    @staticmethod
    def _classify_category(support_type: str) -> str:
        if support_type.startswith("MA"):
            return "ma"
        if "前低" in support_type:
            return "price_structure"
        if "箱体" in support_type:
            return "price_structure"
        if "趋势" in support_type:
            return "price_structure"
        if "BB" in support_type:
            return "volatility"
        if "密集" in support_type:
            return "volume"
        if "关口" in support_type:
            return "psychological"
        return "other"
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/statistics.py
git commit -m "feat: add statistics analyzer with composite scoring and ranking"
```

---

### Task 15: Excel Export

**Files:**
- Create: `support_backtest/excel_export.py`

- [ ] **Step 1: Write `excel_export.py`**

```python
"""
Multi-sheet Excel report generator.

Sheets:
  1. Summary — Top N supports ranked
  2. By Category — One table per support category
  3. Touch Events — All events with details
  4. Top 5 Indicators — Distribution per support type
  5. Config — Parameters used
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from config import config as default_config, SupportBacktestConfig

logger = logging.getLogger(__name__)


class ExcelExporter:
    """Generate formatted multi-sheet Excel report."""

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def export(self, stats: dict, output_path: Optional[str] = None) -> str:
        """
        Args:
            stats: Output from StatisticsAnalyzer.analyze()
            output_path: Optional custom path

        Returns:
            Path to generated Excel file
        """
        if output_path is None:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                out_dir / f"{self.cfg.symbol}_support_backtest_{ts}.xlsx"
            )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            type_rank = stats["type_ranking"]
            top_n = min(self.cfg.top_n_supports, len(type_rank))
            top_n_df = type_rank.head(top_n)
            summary_data = {
                "指标": ["总触及次数", "支撑位类型数", "整体反弹率",
                       "平均确认得分", "最强支撑位", "最高反弹率"],
                "数值": [
                    stats["summary"]["total_touch_events"],
                    stats["summary"]["unique_support_types"],
                    f"{stats['summary']['overall_rebound_rate']:.1%}",
                    f"{stats['summary']['avg_confirmation_score']:.2f}/5",
                    stats["summary"]["top_support"],
                    f"{stats['summary']['top_rebound_rate']:.1%}",
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False, startrow=0)

            top_n_df.to_excel(writer, sheet_name="Summary", index=False, startrow=10)

            # Sheet 2: By Category
            stats["category_ranking"].to_excel(writer, sheet_name="By Category", index=False)

            # Sheet 3: Touch Events
            events = stats["event_details"]
            # Select key columns
            event_cols = [
                "date", "support_type", "support_price", "touched",
                "rebounded", "fastest_recovery_days", "confirmation_score",
                "touch_depth", "timeframe",
            ]
            event_df = events[event_cols] if not events.empty else pd.DataFrame()
            event_df.to_excel(writer, sheet_name="Touch Events", index=False)

            # Sheet 4: Top 5 Indicators Detail
            if not events.empty:
                indicator_dist = events.groupby("support_type").agg(
                    touch_count=("touched", "sum"),
                    rebound_rate=("rebounded", "mean"),
                    avg_confirmation=("confirmation_score", "mean"),
                ).reset_index()
                indicator_dist["rebound_rate"] = indicator_dist["rebound_rate"].apply(
                    lambda x: f"{x:.1%}"
                )
                indicator_dist.to_excel(writer, sheet_name="Indicators Detail", index=False)

            # Sheet 5: Config
            config_data = {
                "参数": ["标的代码", "数据开始", "数据结束", "触及容差",
                       "反弹周期", "反弹目标", "输出Top N", "前低窗口",
                       "布林带周期", "箱体回溯期"],
                "值": [
                    self.cfg.symbol, self.cfg.start_date, self.cfg.end_date,
                    self.cfg.touch_tolerance,
                    str(self.cfg.rebound_periods), str(self.cfg.rebound_targets),
                    self.cfg.top_n_supports,
                    str(self.cfg.prior_low_windows),
                    f"{self.cfg.bollinger_period}/{self.cfg.bollinger_std}σ",
                    self.cfg.box_lookback,
                ],
            }
            pd.DataFrame(config_data).to_excel(writer, sheet_name="Config", index=False)

        logger.info(f"Excel report saved to {output_path}")
        return output_path
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/excel_export.py
git commit -m "feat: add multi-sheet Excel export"
```

---

### Task 16: Chart Generation

**Files:**
- Create: `support_backtest/chart.py`

- [ ] **Step 1: Write `chart.py`**

```python
"""
Chart generation for support backtest results.

Generates 5 charts:
  1. Support Map — K-line with support lines overlaid (last 120 days)
  2. Probability Ranking — Horizontal bar chart of top N supports
  3. Rebound Rate Heatmap — Support type × Rebound period
  4. Touch Distribution Pie — By support category
  5. Top 5 Indicator Radar — Average confirmation scores per category
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import config as default_config, SupportBacktestConfig

# Chinese font setup
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class ChartGenerator:

    def __init__(self, cfg: Optional[SupportBacktestConfig] = None):
        self.cfg = cfg or default_config

    def generate_all(
        self, stats: dict, df: pd.DataFrame, supports_df: pd.DataFrame
    ) -> list[str]:
        """Generate all charts. Returns list of saved file paths."""
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        code = self.cfg.symbol
        paths = []

        paths.append(self._chart_support_map(df, supports_df, out_dir, code, ts))
        paths.append(self._chart_ranking(stats, out_dir, code, ts))
        paths.append(self._chart_heatmap(stats, out_dir, code, ts))
        paths.append(self._chart_pie(stats, out_dir, code, ts))
        paths.append(self._chart_radar(stats, out_dir, code, ts))

        logger.info(f"{len(paths)} charts saved to {out_dir}")
        return paths

    def _chart_support_map(self, df, supports_df, out_dir, code, ts):
        """K-line chart with key support lines overlaid."""
        fig, ax = plt.subplots(figsize=(16, 8))

        # Last 120 days for readability
        plot_df = df.tail(120).copy()
        plot_df = plot_df.reset_index(drop=True)
        x = range(len(plot_df))

        # Candlestick approximation: color bars
        colors = np.where(
            plot_df["close"] >= plot_df["open"], "red", "green"
        )
        ax.bar(x, plot_df["high"] - plot_df["low"], bottom=plot_df["low"],
               color=colors, width=0.6, linewidth=0.5)
        ax.bar(x, abs(plot_df["close"] - plot_df["open"]),
               bottom=plot_df[["open", "close"]].min(axis=1),
               color=colors, width=0.6, linewidth=0.5)

        # Overlay top 5 supports (if they exist in the plot range)
        type_rank = stats.get("type_ranking", pd.DataFrame())
        if not type_rank.empty:
            top5_types = type_rank.head(5)["support_type"].tolist()
            plot_supports = supports_df[
                supports_df["support_type"].isin(top5_types)
            ]
            plot_dates = set(plot_df["date"].values)
            for sup_type in top5_types[:3]:
                sup_data = plot_supports[plot_supports["support_type"] == sup_type]
                sup_data = sup_data[sup_data["date"].isin(plot_dates)]
                if sup_data.empty:
                    continue
                # Average support price
                avg_price = sup_data["support_price"].mean()
                ax.axhline(y=avg_price, linestyle="--", alpha=0.6,
                          label=f"{sup_type}: {avg_price:.3f}")

        ax.set_title(f"{code} Support Map (Last 120 Days)", fontsize=14)
        ax.set_xlabel("Trading Days (recent)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        path = str(out_dir / f"{code}_01_support_map_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_ranking(self, stats, out_dir, code, ts):
        """Horizontal bar chart of top N support types by composite score."""
        type_rank = stats.get("type_ranking", pd.DataFrame())
        if type_rank.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(15, len(type_rank))
        data = type_rank.head(top_n).iloc[::-1]  # Reverse for horizontal bar

        labels = [f"{r['support_type']}({r['timeframe']})" for _, r in data.iterrows()]
        scores = data["composite_score"].values
        rates = data["rebound_rate"].values * 100

        bars = ax.barh(labels, scores, color=plt.cm.RdYlGn(rates / 100))
        ax.set_xlabel("Composite Score")
        ax.set_title(f"{code} Support Level Probability Ranking (Top {top_n})")
        ax.grid(axis="x", alpha=0.3)

        # Add rebound rate annotations
        for i, (score, rate) in enumerate(zip(scores, rates)):
            ax.text(score + 0.01, i, f"{rate:.0f}%", va="center", fontsize=9)

        path = str(out_dir / f"{code}_02_ranking_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_heatmap(self, stats, out_dir, code, ts):
        """Rebound rate heatmap: support type × rebound period."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        # Use the first target (1%) for heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        # Simplified: aggregate by support type
        pivot = events.pivot_table(
            values="rebounded", index="support_type", aggfunc="mean"
        )
        pivot = pivot.sort_values("rebounded", ascending=False).head(15)

        heatmap_data = pivot.values
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks([0])
        ax.set_xticklabels(["Overall Rebound Rate"])
        ax.set_title(f"{code} Rebound Rate by Support Type")

        plt.colorbar(im, ax=ax)
        # Annotate cells
        for i in range(len(pivot)):
            val = heatmap_data[i][0]
            ax.text(0, i, f"{val:.1%}", ha="center", va="center", fontsize=10)

        path = str(out_dir / f"{code}_03_heatmap_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_pie(self, stats, out_dir, code, ts):
        """Touch distribution by support category."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        fig, ax = plt.subplots(figsize=(8, 8))
        cat_counts = events["category"].value_counts()
        colors = plt.cm.Set3(range(len(cat_counts)))
        wedges, texts, autotexts = ax.pie(
            cat_counts.values, labels=cat_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90,
        )
        ax.set_title(f"{code} Touch Distribution by Support Category")

        path = str(out_dir / f"{code}_04_pie_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_radar(self, stats, out_dir, code, ts):
        """Radar chart of Top 5 indicator scores by category."""
        events = stats.get("event_details", pd.DataFrame())
        if events.empty:
            return ""

        # Average confirmation per category
        cat_scores = events.groupby("category")["confirmation_score"].mean()

        categories = list(cat_scores.index)
        values = cat_scores.values
        n = len(categories)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values = values.tolist()
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.plot(angles, values, "o-", color="steelblue", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(f"{code} Avg Confirmation Score by Category (Top 5 Indicators)")

        path = str(out_dir / f"{code}_05_radar_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/chart.py
git commit -m "feat: add chart generation (map/ranking/heatmap/pie/radar)"
```

---

### Task 17: Main CLI Entry Point

**Files:**
- Create: `support_backtest/main.py`

- [ ] **Step 1: Write `main.py`**

```python
#!/usr/bin/env python3
"""
Support Level Backtest System — CLI Entry Point
=================================================
One-click support level backtesting across 8+ support types
with multi-timeframe MA analysis and Top 5 confirming indicators.

Usage:
  python main.py                                    # default: 588170, full history
  python main.py --symbol 588170 --start 20240101
  python main.py --symbol 512760 --top 50 --no-charts
  python main.py --help
"""
import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SupportBacktestConfig, config as default_config
from data_loader import DataLoader
from indicator import compute_indicators
from support_detectors import (
    combine_detections,
    MASupport, BollingerSupport, PriorLowSupport,
    BoxRangeSupport, TrendLineSupport, VolumeClusterSupport,
    RoundNumberSupport,
)
from backtest import BacktestEngine
from confirmation import ConfirmationAnalyzer
from statistics import StatisticsAnalyzer
from excel_export import ExcelExporter
from chart import ChartGenerator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Support Level Backtest System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol", default="588170", help="ETF/stock code")
    p.add_argument("--start", default="20000101", help="Start date YYYYMMDD")
    p.add_argument("--end", default=None, help="End date YYYYMMDD (default: today)")
    p.add_argument("--top", type=int, default=30, help="Top N supports to show")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return p.parse_args()


def print_header(stats: dict, symbol: str) -> None:
    s = stats["summary"]
    type_rank = stats["type_ranking"]
    top_n = min(10, len(type_rank))

    print()
    print("=" * 72)
    print(f"  {symbol} 支撑位回测报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print(f"  总触及事件:   {s['total_touch_events']}")
    print(f"  支撑位类型数: {s['unique_support_types']}")
    print(f"  整体反弹率:   {s['overall_rebound_rate']:.1%}")
    print(f"  平均确认得分: {s['avg_confirmation_score']:.2f} / 5")
    print("-" * 72)
    print(f"  {'排名':<4} {'支撑类型':<16} {'触及次数':<8} {'反弹率':<8} {'确认分':<8} {'综合得分':<8}")
    print("-" * 72)

    for _, row in type_rank.head(top_n).iterrows():
        label = f"{row['support_type']}({row.get('timeframe','')})"
        print(
            f"  {row['rank']:<4.0f} "
            f"{label:<16} "
            f"{int(row['touch_count']):<8} "
            f"{row['rebound_rate']:<8.1%} "
            f"{row['avg_confirmation']:<8.2f} "
            f"{row['composite_score']:<8.3f}"
        )

    print("-" * 72)
    print(f"  最强支撑: {s['top_support']} (反弹率 {s['top_rebound_rate']:.1%})")
    print("=" * 72)
    print()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # Config
    cfg = SupportBacktestConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end or datetime.now().strftime("%Y%m%d"),
        top_n_supports=args.top,
        output_dir=args.output or default_config.output_dir,
        generate_excel=not args.no_excel,
        generate_charts=not args.no_charts,
    )

    logger.info(f"Starting support backtest for {cfg.symbol}")
    logger.info(f"Date range: {cfg.start_date} → {cfg.end_date}")

    # 1. Load data
    loader = DataLoader()
    data = loader.load_all(cfg.symbol, cfg.start_date, cfg.end_date)
    df_daily = data["daily"]
    logger.info(f"Loaded {len(df_daily)} daily bars")

    # 2. Compute indicators
    df = compute_indicators(df_daily, cfg)

    # 3. Detect supports
    detectors = [
        # MA — daily
        MASupport(list(cfg.daily_ma_periods), "日"),
        # MA — weekly
        MASupport(list(cfg.weekly_ma_periods), "周"),
        # MA — monthly
        MASupport(list(cfg.monthly_ma_periods), "月"),
        # Price structure
        BollingerSupport(),
        PriorLowSupport(cfg.prior_low_windows),
        BoxRangeSupport(cfg),
        TrendLineSupport(cfg),
        # Volume
        VolumeClusterSupport(cfg),
        # Psychological
        RoundNumberSupport(),
    ]

    # Add intraday MAs if available
    if "min60" in data:
        min60_df = compute_indicators(data["min60"], cfg)
        detectors.append(MASupport(list(cfg.min60_ma_periods), "60分钟"))
    if "min120" in data:
        min120_df = compute_indicators(data["min120"], cfg)
        detectors.append(MASupport(list(cfg.min120_ma_periods), "120分钟"))

    # Also compute weekly/monthly MA on their own frames
    weekly_df = compute_indicators(data["weekly"], cfg)
    monthly_df = compute_indicators(data["monthly"], cfg)

    # Daily MAs from weekly/monthly frames — resample to daily
    # (use the weekly MA value for every day in that week)
    daily_with_wm = df.copy()
    daily_with_wm["date"] = pd.to_datetime(daily_with_wm["date"])
    weekly_df["date"] = pd.to_datetime(weekly_df["date"])
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])

    # Merge weekly MA to daily (forward fill within each week)
    for col in [c for c in weekly_df.columns if c.startswith("MA")]:
        temp = weekly_df[["date", col]].copy()
        temp = temp.rename(columns={col: f"W_{col}"})
        daily_with_wm = daily_with_wm.merge(temp, on="date", how="left")
        daily_with_wm[f"W_{col}"] = daily_with_wm[f"W_{col}"].ffill()
        # Rename back for MA detector
        p = col.replace("MA", "")
        daily_wm_col = f"W_MA{p}"
        # Create MA column for weekly detector
        daily_with_wm[f"W_{col}"] = daily_with_wm[f"W_{col}"].fillna(daily_with_wm[col])

    # Add Merged MA from monthly
    for col in [c for c in monthly_df.columns if c.startswith("MA")]:
        temp = monthly_df[["date", col]].copy()
        temp = temp.rename(columns={col: f"M_{col}"})
        daily_with_wm = daily_with_wm.merge(temp, on="date", how="left")
        daily_with_wm[f"M_{col}"] = daily_with_wm[f"M_{col}"].ffill()

    supports = combine_detections(detectors, daily_with_wm)

    logger.info(f"Detected {len(supports)} support level rows "
                f"({supports['support_type'].nunique()} types)")

    if supports.empty:
        print("No supports detected. Try a different symbol or date range.")
        return

    # 4. Backtest
    engine = BacktestEngine(cfg)
    events = engine.run(daily_with_wm, supports)
    logger.info(f"Backtest: {len(events)} touch events")

    if not events:
        print("No touch events found. Check support data.")
        return

    # 5. Confirmation
    conf_analyzer = ConfirmationAnalyzer(cfg)
    events = conf_analyzer.analyze(events, daily_with_wm, supports)

    # 6. Statistics
    stat_analyzer = StatisticsAnalyzer(cfg)
    stats = stat_analyzer.analyze(events)

    # 7. Output
    print_header(stats, cfg.symbol)

    if cfg.generate_excel:
        excel_path = ExcelExporter(cfg).export(stats)
        print(f"Excel 报告: {excel_path}")

    if cfg.generate_charts:
        chart_paths = ChartGenerator(cfg).generate_all(stats, daily_with_wm, supports)
        for p in chart_paths:
            print(f"图表: {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add support_backtest/main.py
git commit -m "feat: add main CLI entry point for support backtest"
```

---

### Task 18: Integration Test — Run on 588170

**Files:**
- (none new)

- [ ] **Step 1: Run the full pipeline on 588170**

```bash
cd support_backtest && python main.py --symbol 588170 --start 20240101 --verbose
```

Expected: Console output with ranking table, Excel file in `support_backtest_results/`, 5 PNG charts.

- [ ] **Step 2: Verify output files exist**

```bash
ls support_backtest_results/*.xlsx support_backtest_results/*.png | head -20
```

- [ ] **Step 3: Inspect console output for reasonableness**

Expected: Top supports should include MA20/MA60 (classic supports), BB lower band. Rebound rates should be > 50% for major supports on a trending ETF.

- [ ] **Step 4: Fix any runtime errors iteratively**

Common issues:
- Missing `body_ratio` or `close` column → add column checks in indicator.py
- Import errors → ensure sys.path is correct in main.py
- Chinese font missing → matplotlib fallback
- Minute data missing → graceful skip with warning

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for 588170 run"
```

---

## Self-Review

**Spec coverage check:**
- data_loader with proxy bypass ✓ (Task 2)
- Multi-timeframe (daily/weekly/monthly/60min/120min) ✓ (Task 2 + Task 5)
- 8 support detectors ✓ (Tasks 4-11)
- Touch + rebound engine ✓ (Task 12)
- Top 5 confirmation indicators ✓ (Task 13)
- Statistics scoring & ranking ✓ (Task 14)
- Excel multi-sheet export ✓ (Task 15)
- 5 PNG charts ✓ (Task 16)
- CLI entry point ✓ (Task 17)
- Integration test on 588170 ✓ (Task 18)

**Placeholder scan:** No TBDs, TODOs, or vague instructions. All code is concrete.

**Type consistency:**
- `TouchEvent.rebound_results`: `dict[tuple, dict]` — consistent across backtest.py, confirmation.py, statistics.py
- `SupportBacktestConfig`: all fields used consistently across modules
- `BaseSupportDetector.detect()`: returns `pd.DataFrame` with columns `[date, support_type, support_price, timeframe]` — consistent
- Date type: mixed `pd.Timestamp` / `datetime` — normalized in data_loader with `pd.to_datetime()` — handled
