"""
Trading simulator — backtest actual trading rules on historical signals.

Rules (all configurable):
  - Buy at close on signal day
  - ATR-based or fixed % stop loss
  - Trailing take profit
  - Max hold period (default 10 days)
  - Only one position at a time
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from ma_config import config as default_config

logger = logging.getLogger(__name__)


class TradingSimulator:
    """Simulate trades with configurable rules."""

    def __init__(self, capital: float = 60000, position_pct: float = 0.20,
                 stop_loss: float = None,          # None = ATR-based
                 take_profit: float = 0.05,
                 max_hold: int = 10,
                 trail_stop: float = 0.03):
        self.capital = capital
        self.position_pct = position_pct
        self.stop_loss = stop_loss      # fixed % (None = use ATR)
        self.take_profit = take_profit
        self.max_hold = max_hold
        self.trail_stop = trail_stop

    def run(self, close: pd.Series, bottom_prob: pd.Series,
            entry_threshold: float = 70,
            atr_series: pd.Series = None,
            atr_multiple: float = 2.0) -> dict:
        """
        Run simulation on historical data.

        Args:
            close: daily close prices
            bottom_prob: model's bottom probability (0-100)
            entry_threshold: enter when prob >= this value
            atr_series: ATR(14) values (required if stop_loss=None)
            atr_multiple: multiplier for ATR-based stop (default 2.0)
        """
        close = np.asarray(close, dtype=float)
        prob = np.asarray(bottom_prob, dtype=float)
        n = len(close)

        if atr_series is not None:
            atr = np.asarray(atr_series, dtype=float)
        else:
            # Compute ATR on the fly
            atr = np.zeros(n)
            tr = np.maximum(
                np.abs(close[1:] - close[:-1]),
                np.maximum(np.abs(close[1:] - close[:-1]), 0)
            )
            for i in range(14, n):
                atr[i] = np.mean(tr[i-13:i+1])
            atr[:14] = atr[14] if n > 14 else close[0] * 0.03

        cash = self.capital
        position = 0.0
        entry_price = 0.0
        entry_idx = -1
        peak_since_entry = 0.0
        trades = []
        equity = np.zeros(n)

        for i in range(n):
            price = close[i]

            if position > 0:
                peak_since_entry = max(peak_since_entry, price)
                entry_stop = entry_price * (1 - self.stop_loss) if self.stop_loss else \
                             entry_price - atr_multiple * atr[i]
                days_held = i - entry_idx

                # Exit conditions: SL, TP, max hold, trail
                exit_reason = None
                exit_price = price

                if price <= entry_stop:
                    exit_reason = 'stop_loss'
                elif price >= entry_price * (1 + self.take_profit):
                    exit_reason = 'take_profit'
                elif days_held >= self.max_hold:
                    exit_reason = 'max_hold'
                elif price <= peak_since_entry * (1 - self.trail_stop):
                    exit_reason = 'trail_stop'

                if exit_reason:
                    pnl_pct = (price / entry_price - 1) * 100
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_price': entry_price, 'exit_price': price,
                        'pnl': position * (price - entry_price),
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                    })
                    cash += position * price
                    position = 0
                    entry_price = 0
                    entry_idx = -1
                    peak_since_entry = 0.0

            # Entry
            if position == 0 and prob[i] >= entry_threshold:
                position_size = cash * self.position_pct
                entry_price = price
                entry_idx = i
                position = position_size / price
                peak_since_entry = price
                cash -= position_size

            equity[i] = cash + position * price

        return self._compute_metrics(trades, equity, close, self.capital)

    def _compute_metrics(self, trades: list, equity: np.ndarray,
                          close: np.ndarray, capital: float) -> dict:
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'annual_return': 0,
                    'max_drawdown': 0, 'avg_return': 0, 'consecutive_losses': 0,
                    'sharpe_ratio': 0, 'equity_curve': equity, 'trades': []}

        pnls_pct = [t['pnl'] / capital * 100 for t in trades]
        wins = sum(1 for p in pnls_pct if p > 0)
        win_rate = wins / len(trades) * 100

        max_consec = 0; current = 0
        for p in pnls_pct:
            if p <= 0: current += 1; max_consec = max(max_consec, current)
            else: current = 0

        total_ret = (equity[-1] / equity[0] - 1) * 100
        days = len(close)
        ann_ret = ((1 + total_ret / 100) ** (252 / max(days, 1)) - 1) * 100
        peak = np.maximum.accumulate(equity)
        max_dd = abs(np.min((equity - peak) / peak * 100))
        daily_r = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(252) * np.mean(daily_r) / np.std(daily_r) if np.std(daily_r) > 0 else 0

        return {
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'annual_return': round(ann_ret, 1),
            'max_drawdown': round(max_dd, 1),
            'avg_return': round(np.mean(pnls_pct), 1),
            'consecutive_losses': max_consec,
            'sharpe_ratio': round(sharpe, 2),
            'equity_curve': equity,
            'trades': trades,
        }
