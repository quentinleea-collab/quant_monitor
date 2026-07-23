"""
Buy-point entry detector — finds optimal entry timing within a bottom zone.

Uses: volume contraction/expansion, intraday (60min/120min) patterns,
candlestick reversal signals, multi-timeframe alignment.

Output: Entry Score (0-100), suggested entry/slop levels, progressive stops.
"""
import sys, os, logging
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'support_backtest'))
from data_loader import DataLoader
from indicator import compute_indicators

logger = logging.getLogger(__name__)


class EntryDetector:
    """
    Detect optimal buy point within a confirmed bottom zone.

    Scoring components (weighted):
      - Volume shrinkage-to-expansion: 25 pts
      - 60min MACD/RSI turning: 25 pts (15 if no intraday data)
      - 120min trend alignment: 15 pts
      - K-line reversal pattern: 15 pts
      - Support level proximity: 10 pts
      - Multi-TF confirmation: 10 pts
    """

    def __init__(self):
        self.loader = DataLoader()
        self._has_intraday = False

    def analyze(self, symbol: str, start: str = None, end: str = None,
                bottom_prob: float = None) -> dict:
        """
        Analyze buy-point timing for a symbol.

        Returns dict with:
          entry_score, entry_price, stop_initial, stop_breakeven, stop_trail,
          volume_signal, intraday_signal, pattern_signal, support_signal,
          recommendation, confidence
        """
        # ── Load daily data ──────────────────────────────────
        daily = self.loader.fetch_daily(symbol, start or '20240101', end or '20260722')
        if daily.empty:
            return self._empty_result("无日线数据")
        daily = daily.sort_values('date').reset_index(drop=True)
        daily = compute_indicators(daily)

        last = daily.iloc[-1]
        current_price = float(last['close'])

        # ── Try intraday data ────────────────────────────────
        min60 = self._safe_fetch_minute(symbol)
        min120 = None
        if min60 is not None and len(min60) >= 4:
            self._has_intraday = True
            # Resample 60min → 120min
            min120 = self._resample_120(min60)

        # ── 1. Volume signal (25 pts) ────────────────────────
        vol_score, vol_detail = self._score_volume(daily)

        # ── 2. Intraday signal (25 pts) ──────────────────────
        intra_score, intra_detail = self._score_intraday(min60, min120, current_price)

        # ── 3. K-line pattern (15 pts) ───────────────────────
        pattern_score, pattern_detail = self._score_pattern(last)

        # ── 4. Support proximity (10 pts) ────────────────────
        support_score, support_detail = self._score_support(daily, current_price)

        # ── 5. Multi-TF alignment (10 pts) ───────────────────
        multitf_score, multitf_detail = self._score_multitf(daily, min60)

        # ── 6. Bottom context bonus (15 pts) ──────────────────
        context_bonus = 15 if (bottom_prob and bottom_prob >= 70) else (10 if bottom_prob and bottom_prob >= 50 else 0)

        # ── Aggregate ────────────────────────────────────────
        total = vol_score + intra_score + pattern_score + support_score + multitf_score + context_bonus

        # Confidence based on data quality
        confidence = '高' if self._has_intraday else '中(无分时数据)'

        # Recommendation
        if total >= 70:
            rec = '强烈买入信号'
        elif total >= 55:
            rec = '买入信号'
        elif total >= 40:
            rec = '观察(等待确认)'
        else:
            rec = '继续等待'

        # Progressive stop levels
        stops = self._calc_progressive_stops(current_price, support_detail.get('nearest_support'), daily)

        return {
            'entry_score': total,
            'entry_price': round(current_price, 3),
            'stop_initial': stops['initial'],
            'stop_breakeven': stops['breakeven'],
            'stop_trail_tight': stops['trail_tight'],
            'volume_signal': vol_detail,
            'intraday_signal': intra_detail,
            'pattern_signal': pattern_detail,
            'support_signal': support_detail,
            'multitf_signal': multitf_detail,
            'bottom_context': f'底部概率{bottom_prob:.0f}%' if bottom_prob else '未评估',
            'recommendation': rec,
            'confidence': confidence,
        }

    # ═══ Scoring Components ═══════════════════════════════════════════

    def _score_volume(self, daily: pd.DataFrame) -> tuple[int, dict]:
        """Volume: shrinkage then expansion = buy signal."""
        last = daily.iloc[-1]
        vol_ratio_5 = float(last.get('vol_ma5', 1))
        vol_ratio_20 = float(last.get('vol_ma20', 1))
        vol = float(last['volume'])

        # Check recent volume trend (last 5 days)
        recent_vol = daily['volume'].iloc[-5:].values
        vol_ma5_val = daily['volume'].rolling(5).mean().iloc[-1]
        vol_shrinking = all(recent_vol[i] <= recent_vol[i+1] * 1.05 for i in range(len(recent_vol)-1))
        vol_expanding = vol > vol_ma5_val * 1.1  # today > 110% of 5-day avg

        detail = {}
        score = 0

        if vol / vol_ratio_5 < 0.6:
            detail['shrinkage'] = '极度缩量'
            score += 12
        elif vol / vol_ratio_5 < 0.8:
            detail['shrinkage'] = '缩量'
            score += 8

        if vol_expanding:
            detail['expansion'] = '放量'
            score += 10
            if vol_shrinking:
                detail['pattern'] = '缩量后放量(经典底部量价)'
                score += 3
        else:
            detail['expansion'] = '未放量'

        detail['vol_ratio'] = round(float(vol / vol_ratio_5), 2)
        return score, detail

    def _score_intraday(self, min60, min120, current_price) -> tuple[int, dict]:
        """60min/120min MACD and RSI turning points."""
        if min60 is None or len(min60) < 8:
            return 8, {'status': '无分时数据, 默认中性分', 'fallback': True}

        score = 0
        detail = {}

        # 60min MACD histogram direction
        if 'close' in min60.columns:
            macd_hist = self._macd_hist(min60['close'])
            if len(macd_hist) >= 4:
                recent = macd_hist.iloc[-4:].values
                if recent[-1] > recent[-2] and recent[-2] < recent[-3]:
                    detail['m60_macd'] = 'MACD绿柱缩短(拐头)'
                    score += 8
                elif recent[-1] > 0:
                    detail['m60_macd'] = 'MACD红柱'
                    score += 12
                else:
                    detail['m60_macd'] = 'MACD绿柱'
                    score += 2

            # 60min RSI
            rsi = self._rsi(min60['close'], 14)
            if len(rsi) >= 2:
                rsi_now = float(rsi.iloc[-1])
                detail['m60_rsi'] = round(rsi_now, 1)
                if rsi_now < 30:
                    score += 8
                    detail['m60_rsi_signal'] = '超卖'
                elif rsi_now < 40 and float(rsi.iloc[-2]) < rsi_now:
                    score += 5
                    detail['m60_rsi_signal'] = '脱离超卖区'

        # 120min alignment
        if min120 is not None and len(min120) >= 3 and 'close' in min120.columns:
            rsi120 = self._rsi(min120['close'], 14)
            if len(rsi120) >= 2:
                r120_now = float(rsi120.iloc[-1])
                detail['m120_rsi'] = round(r120_now, 1)
                if r120_now < 40:
                    score += 5
                    detail['m120_rsi_signal'] = '接近超卖'
                elif r120_now < 50 and float(rsi120.iloc[-2]) < r120_now:
                    score += 3
                    detail['m120_rsi_signal'] = '拐头向上'

        return min(score, 25), detail

    def _score_pattern(self, last: pd.Series) -> tuple[int, dict]:
        """K-line reversal patterns."""
        score = 0
        detail = {}

        lower_shadow = float(last.get('lower_shadow', 0))
        body_ratio = float(last.get('body_ratio', 0))
        is_hammer = bool(last.get('pat_hammer', False))
        is_bullish = bool(last.get('pat_bullish_engulf', False))
        is_morning = bool(last.get('pat_morning_star', False))
        is_piercing = bool(last.get('pat_piercing', False))

        if lower_shadow >= 0.6:
            detail['long_shadow'] = f'长下影({lower_shadow:.0%})'
            score += 7
        if is_hammer:
            detail['hammer'] = '锤子线'
            score += 5
        if is_bullish:
            detail['bullish'] = '看涨吞没'
            score += 6
        if is_morning:
            detail['morning'] = '启明星'
            score += 8
        if is_piercing:
            detail['piercing'] = '刺透形态'
            score += 4

        if body_ratio > 0.5 and float(last['close']) > float(last['open']):
            detail['strong_body'] = '强势阳线'

        return min(score, 15), detail

    def _score_support(self, daily: pd.DataFrame, current_price: float) -> tuple[int, dict]:
        """Proximity to key support levels."""
        score = 0
        detail = {}

        close = daily['close'].values
        low = daily['low'].values

        # Find nearest swing low
        swing_lows = []
        for i in range(15, len(close) - 5):
            if low[i] <= np.min(low[i-15:i]) and low[i] <= np.min(low[i+1:i+6]):
                swing_lows.append(float(low[i]))

        nearest = None
        for sl in sorted(swing_lows, reverse=True):
            if sl < current_price * 0.99:
                nearest = sl
                break

        if nearest:
            distance = (current_price - nearest) / current_price * 100
            detail['nearest_support'] = round(nearest, 2)
            detail['distance_pct'] = round(distance, 1)

            if distance < 3:
                score = 10
                detail['level'] = '极近支撑(最优买点)'
            elif distance < 5:
                score = 7
                detail['level'] = '接近支撑'
            elif distance < 8:
                score = 4
                detail['level'] = '距支撑适中'
            else:
                score = 1

        # Also check MA60
        ma60 = float(daily['MA60'].iloc[-1]) if 'MA60' in daily.columns and not pd.isna(daily['MA60'].iloc[-1]) else None
        if ma60:
            detail['ma60'] = round(ma60, 2)
            ma60_dist = (current_price - ma60) / current_price * 100
            if ma60_dist < 0 and abs(ma60_dist) < 5:
                score += 3

        return min(score, 10), detail

    def _score_multitf(self, daily: pd.DataFrame, min60) -> tuple[int, dict]:
        """Multi-timeframe alignment check."""
        score = 0
        detail = {}

        # Check if daily RSI is turning up
        if 'RSI6' in daily.columns:
            rsi = daily['RSI6'].values[-5:]
            if len(rsi) >= 3:
                if rsi[-1] > rsi[-2] and rsi[-2] < rsi[-3]:
                    detail['daily_rsi'] = 'RSI拐头'
                    score += 4

        # Check MACD histogram direction
        hist = self._macd_hist(daily['close'])
        if len(hist) >= 3:
            h = hist.iloc[-3:].values
            if h[-1] > h[-2]:  # green bar growing or red bar shrinking
                detail['daily_macd'] = 'MACD改善'
                score += 3

        # 60min aligned with daily?
        if min60 is not None and len(min60) >= 8:
            detail['tf_alignment'] = '日线+60分钟联合分析可用'
            score += 3
        else:
            detail['tf_alignment'] = '仅日线'

        return min(score, 10), detail

    # ═══ Progressive Stops ═══════════════════════════════════════════

    def _calc_progressive_stops(self, entry: float, support: float = None,
                                 daily: pd.DataFrame = None) -> dict:
        """
        3-stage progressive stops.
        Stage 1: Initial — tighter stop based on recent low (not ultimate support)
        Stage 2: Breakeven — after +3% move stop to entry
        Stage 3: Trailing — after +5%, trail from peak at -2%
        """
        # Stage 1: use recent 5-day low for tighter entry stop
        if daily is not None and len(daily) >= 5:
            recent_low = float(daily['low'].iloc[-5:].min())
            initial = round(recent_low * 0.99, 3)  # 1% below recent low
        elif support and support < entry:
            initial = round(support * 0.98, 3)
        else:
            initial = round(entry * 0.97, 3)

        dist_pct = (entry - initial) / entry * 100

        return {
            'initial': f'{initial:.3f} (-{dist_pct:.1f}%, 近5日低点下方)',
            'breakeven': f'{entry:.3f} (触发+3%后止损移至成本价)',
            'trail_tight': f'触发+5%后, 从最高点回落-2%全平',
        }

    # ═══ Helpers ═════════════════════════════════════════════════════

    def _safe_fetch_minute(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try to get 60min data, return None if unavailable."""
        try:
            df = self.loader.fetch_minute(symbol, '60', '20260101', '20260722')
            if df is not None and len(df) >= 4:
                return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logger.debug(f"No 60min data for {symbol}: {e}")
        return None

    @staticmethod
    def _resample_120(min60: pd.DataFrame) -> pd.DataFrame:
        """Resample 60min to 120min bars."""
        df = min60.set_index('date').sort_index()
        result = df.resample('2h').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        return result.reset_index()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        return 2 * (dif - dea)

    @staticmethod
    def _empty_result(reason: str) -> dict:
        return {
            'entry_score': 0, 'entry_price': 0,
            'stop_initial': 'N/A', 'stop_breakeven': 'N/A', 'stop_trail_tight': 'N/A',
            'recommendation': reason, 'confidence': '无',
        }
