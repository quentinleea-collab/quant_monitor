"""
Exit strategy backtest — compares 4 exit models across 3 market regimes.

Models:
  1. Fixed ATR — full exit when price drops entry - N×ATR
  2. MA half + ATR full — sell half at MA20 break, rest at ATR stop
  3. MA cascade — half at MA20, rest at MA60 break with 3-day recovery check
  4. Trailing — after +5%, trail -3% from peak (ATR stop before that)

Regimes: up / sideways / down (classified by MA alignment + 20d return)
"""
import sys, os, logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'support_backtest'))
from data_loader import DataLoader
from indicator import compute_indicators

logger = logging.getLogger(__name__)


@dataclass
class ExitResult:
    model: str
    regime: str
    trades: int
    win_rate: float
    avg_return: float      # per trade, account-level %
    avg_hold_days: float
    max_dd: float           # max portfolio drawdown during this regime
    profit_factor: float    # gross profit / gross loss
    sharpe: float
    exit_reasons: dict = field(default_factory=dict)


class ExitBacktest:
    """Compare exit strategies across regimes."""

    def __init__(self, capital: float = 60000, position_pct: float = 0.20,
                 entry_threshold: float = 70, max_hold: int = 20,
                 atr_multiple: float = 2.0):
        self.capital = capital
        self.position_pct = position_pct
        self.entry_threshold = entry_threshold
        self.max_hold = max_hold
        self.atr_multiple = atr_multiple
        self.loader = DataLoader()

    def compare(self, symbol: str, start: str = '20240101',
                end: str = None) -> pd.DataFrame:
        """Run all 4 models across all regimes, return comparison table."""
        end = end or __import__('datetime').datetime.now().strftime('%Y%m%d')

        # Load data
        daily = self._load_data(symbol, start, end)
        close = daily['close'].values
        high = daily['high'].values
        low = daily['low'].values
        dates = daily['date'].values
        n = len(close)

        # Compute ATR
        atr = self._compute_atr(close, high, low)

        # Compute regime per day
        regimes = self._classify_regimes(daily)

        # Get entry signals (simulated bottom probability)
        # Use actual model if available, else heuristic
        proba = self._get_entry_signals(daily, symbol)

        # Run all 4 models
        results = []
        for model_name in ['Model1_ATR', 'Model2_MAhalf', 'Model3_Cascade', 'Model4_Trail']:
            for regime in ['down', 'sideways', 'up']:
                r = self._run_model(model_name, regime, close, high, low, atr,
                                    regimes, proba, dates, daily)
                if r and r.trades >= 3:
                    results.append(r)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame([{
            '模型': r.model, '行情': r.regime,
            '交易': r.trades, '胜率%': r.win_rate,
            '均收益%': r.avg_return, '均持仓天': r.avg_hold_days,
            '最大回撤%': r.max_dd, '盈亏比': r.profit_factor, 'Sharpe': r.sharpe,
            '退出方式': str(r.exit_reasons)[:80],
        } for r in results])

    def _run_model(self, model: str, regime: str, close, high, low, atr,
                   regimes, proba, dates, daily) -> Optional[ExitResult]:
        """Simulate one model within one regime."""
        n = len(close)
        cash = self.capital
        position = 0.0
        entry_price = 0.0
        entry_idx = -1
        peak = 0.0
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        trades = []
        equity = np.zeros(n)
        exit_reasons = {}

        for i in range(n):
            price = close[i]

            if position > 0:
                peak = max(peak, price)
                days_held = i - entry_idx
                stop_atr = entry_price - self.atr_multiple * atr[i]
                cur_ma20 = ma20[i] if not np.isnan(ma20[i]) else entry_price * 0.9
                cur_ma60 = ma60[i] if not np.isnan(ma60[i]) else entry_price * 0.85
                reason = None

                if model == 'Model1_ATR':
                    if price <= stop_atr or days_held >= self.max_hold:
                        reason = 'ATR' if price <= stop_atr else 'max_hold'

                elif model == 'Model2_MAhalf':
                    # Track if we already halved
                    halved = hasattr(self, '_halved') and self._halved
                    if not halved and price < cur_ma20:
                        cash += (position / 2) * price
                        position /= 2
                        self._halved = True
                    if price <= stop_atr or days_held >= self.max_hold:
                        reason = 'ATR' if price <= stop_atr else 'max_hold'

                elif model == 'Model3_Cascade':
                    halved = getattr(self, '_halved', False)
                    if not halved and price < cur_ma20:
                        cash += (position / 2) * price
                        position /= 2
                        self._halved = True
                    # After halving: full exit if breaks MA60 AND stays below for 3d
                    if halved and price < cur_ma60:
                        # Check if recovered within 3 days (look ahead)
                        recovered = False
                        for j in range(i+1, min(i+4, n)):
                            if close[j] >= ma60[j] if not np.isnan(ma60[j]) else False:
                                recovered = True
                                break
                        if not recovered:
                            reason = 'MA60_cascade'
                    if days_held >= self.max_hold:
                        reason = 'max_hold'

                elif model == 'Model4_Trail':
                    if price >= entry_price * 1.05:
                        # Trailing mode: -3% from peak
                        if price <= peak * 0.97:
                            reason = 'trail'
                    elif price <= stop_atr:
                        reason = 'ATR_pre_trail'
                    if days_held >= self.max_hold:
                        reason = 'max_hold'

                if reason:
                    pnl_pct = (price / entry_price - 1) * 100
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'pnl_pct': pnl_pct,
                        'exit_reason': reason,
                        'days_held': days_held,
                    })
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                    cash += position * price
                    position = 0; entry_price = 0; entry_idx = -1; peak = 0
                    self._halved = False

            # Entry (only in target regime)
            if position == 0 and regimes[i] == regime and proba[i] >= self.entry_threshold:
                pos_size = cash * self.position_pct
                entry_price = price
                entry_idx = i
                position = pos_size / price
                peak = price
                self._halved = False
                cash -= pos_size

            equity[i] = cash + position * price

        if not trades:
            return None

        # Account-level PnL per trade: stock_return% × position%
        pnls_acct = [t['pnl_pct'] * self.position_pct for t in trades]
        wins = sum(1 for p in pnls_acct if p > 0)

        # Max equity drawdown
        peak_eq = np.maximum.accumulate(equity)
        max_dd = float(abs(np.min((equity - peak_eq) / peak_eq * 100)))

        # Profit factor
        gross_win = sum(p for p in pnls_acct if p > 0)
        gross_loss = abs(sum(p for p in pnls_acct if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        return ExitResult(
            model=model.replace('Model1_', '').replace('Model2_', '').replace('Model3_', '').replace('Model4_', ''),
            regime=regime,
            trades=len(trades),
            win_rate=round(wins / len(trades) * 100, 1) if trades else 0,
            avg_return=round(float(np.mean(pnls_acct)), 2),
            avg_hold_days=round(float(np.mean([t['days_held'] for t in trades])), 1),
            max_dd=round(max_dd, 1),
            profit_factor=round(pf, 1),
            sharpe=round(self._sharpe(equity), 2),
            exit_reasons=exit_reasons,
        )

    # ═══ Helpers ═════════════════════════════════════════════════════

    def _load_data(self, symbol, start, end):
        daily = self.loader.fetch_daily(symbol, start, end)
        daily = daily.sort_values('date').reset_index(drop=True)
        daily = compute_indicators(daily)
        return daily

    @staticmethod
    def _compute_atr(close, high, low, period=14):
        tr = np.maximum(high[1:] - low[1:],
               np.maximum(np.abs(high[1:] - close[:-1]),
                          np.abs(low[1:] - close[:-1])))
        atr = np.zeros(len(close))
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        atr[:period] = atr[period] if len(close) > period else close[0] * 0.03
        return atr

    @staticmethod
    def _classify_regimes(daily):
        """Classify each day: up / sideways / down."""
        close = daily['close'].values
        n = len(close)
        regimes = np.full(n, 'sideways', dtype=object)
        for i in range(60, n):
            ret20 = (close[i] / close[i-20] - 1) * 100
            ma20_slope = (pd.Series(close).rolling(20).mean().iloc[i] /
                         pd.Series(close).rolling(20).mean().iloc[i-10] - 1) * 100
            if ret20 > 3 and ma20_slope > 0:
                regimes[i] = 'up'
            elif ret20 < -3 and ma20_slope < 0:
                regimes[i] = 'down'
            else:
                regimes[i] = 'sideways'
        return regimes

    def _get_entry_signals(self, daily, symbol):
        """Get entry signals aligned to daily data length."""
        n = len(daily)
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from feature_engineer import FeatureEngineer
            from model_trainer import ModelTrainer
            fe = FeatureEngineer()
            d = fe.build_features(symbol, '20240101',
                                  __import__('datetime').datetime.now().strftime('%Y%m%d'))
            f = d.drop(columns=['close']) if 'close' in d.columns else d
            mt = ModelTrainer()
            mt.load(symbol, 'rebound_3pct')
            proba_features = mt.predict_proba(f, symbol, 'rebound_3pct') * 100
            # Align to daily: map feature dates to daily dates
            daily_dates = pd.to_datetime(daily['date'].values)
            feat_dates = pd.to_datetime(f.index)
            proba = np.full(n, 50.0)
            for i, dd in enumerate(daily_dates):
                matches = np.where(feat_dates == dd)[0]
                if len(matches) > 0:
                    proba[i] = proba_features[matches[0]]
            return proba
        except Exception:
            pass
        # Heuristic fallback: same length as daily
        close = daily['close'].values
        proba = np.full(n, 50.0)
        rsi = pd.Series(close).diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / \
               (-pd.Series(close).diff().clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi = 100 - 100 / (1 + rsi)
        for i in range(60, n):
            ret20 = (close[i] / close[i-20] - 1) * 100
            ma_dev = (close[i] / pd.Series(close).rolling(60).mean().iloc[i] - 1) * 100
            score = 50 - ret20 * 2 + max(0, 35 - rsi.iloc[i]) - ma_dev * 0.5
            proba[i] = max(0, min(100, score))
        return proba

    @staticmethod
    def _sharpe(equity):
        r = np.diff(equity) / (equity[:-1] + 1e-10)
        return float(np.sqrt(252) * np.mean(r) / np.std(r)) if np.std(r) > 0 else 0
