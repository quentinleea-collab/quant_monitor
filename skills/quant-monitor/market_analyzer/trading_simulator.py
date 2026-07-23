"""
Trading simulator — backtests actual trading rules on historical predictions.

Rules:
  - Capital: 60,000
  - Entry: 20% of capital when bottom_prob >= 70%
  - Stop loss: -3% from entry
  - Take profit: +5%, sell half at this level
  - After TP: if price drops 3% from the peak reached, sell remaining half
  - No new entry while holding a position
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from ma_config import config as default_config

logger = logging.getLogger(__name__)


class TradingSimulator:
    """Simulate trading with user-defined rules on historical signals."""

    def __init__(self, capital: float = 60000, position_pct: float = 0.20,
                 stop_loss: float = -0.03, take_profit: float = 0.05,
                 trail_stop: float = 0.03):
        self.capital = capital
        self.position_pct = position_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trail_stop = trail_stop

    def run(self, close: pd.Series, bottom_prob: pd.Series,
            entry_threshold: float = 70) -> dict:
        """
        Run simulation on historical data.

        Args:
            close: daily close prices
            bottom_prob: model's bottom probability (0-100) per day
            entry_threshold: enter when prob >= this value

        Returns dict with keys:
            trades, equity_curve, annual_return, max_drawdown, win_rate,
            consecutive_losses, total_return, sharpe_ratio
        """
        close = close.values if hasattr(close, 'values') else np.array(close)
        prob = bottom_prob.values if hasattr(bottom_prob, 'values') else np.array(bottom_prob)
        n = len(close)

        cash = self.capital
        position = 0.0
        entry_price = 0.0
        entry_idx = -1       # bar index when position was entered
        peak_since_entry = 0.0
        tp_hit = False
        trades = []
        equity = np.zeros(n)

        for i in range(n):
            price = close[i]

            if position > 0:
                peak_since_entry = max(peak_since_entry, price)

                # Stop loss
                if price <= entry_price * (1 + self.stop_loss):
                    pnl_pct = (price / entry_price - 1) * 100
                    trades.append({
                        'entry_date': entry_idx, 'exit_date': i,
                        'entry_price': entry_price, 'exit_price': price,
                        'pnl': position * (price - entry_price),
                        'pnl_pct': pnl_pct, 'exit_reason': 'stop_loss',
                    })
                    cash += position * price
                    position = 0; entry_price = 0; entry_idx = -1
                    peak_since_entry = 0; tp_hit = False

                # Take profit (sell half at +5%)
                elif not tp_hit and price >= entry_price * (1 + self.take_profit):
                    half = position / 2
                    cash += half * price
                    position -= half
                    tp_hit = True

                # Trail stop (after TP, full exit if drops 3% from peak)
                elif tp_hit and price <= peak_since_entry * (1 - self.trail_stop):
                    pnl_pct = (price / entry_price - 1) * 100
                    trades.append({
                        'entry_date': entry_idx, 'exit_date': i,
                        'entry_price': entry_price, 'exit_price': price,
                        'pnl': position * (price - entry_price),
                        'pnl_pct': pnl_pct, 'exit_reason': 'trail_stop',
                    })
                    cash += position * price
                    position = 0; entry_price = 0; entry_idx = -1
                    peak_since_entry = 0; tp_hit = False

            # Entry signal
            if position == 0 and prob[i] >= entry_threshold:
                position_size = cash * self.position_pct
                entry_price = price
                entry_idx = i
                position = position_size / price
                peak_since_entry = price
                tp_hit = False
                cash -= position_size

            equity[i] = cash + position * price

        # Compute metrics
        return self._compute_metrics(trades, equity, close, self.capital)

    def _compute_metrics(self, trades: list, equity: np.ndarray, close: np.ndarray,
                          capital: float = 60000) -> dict:
        """Compute performance metrics from trade history."""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'annual_return': 0,
                'max_drawdown': 0, 'consecutive_losses': 0,
                'total_return': 0, 'sharpe_ratio': 0,
                'equity_curve': equity,
            }

        # Trade statistics (account-level PnL, not stock-level)
        pnls_pct = [t['pnl'] / capital * 100 for t in trades]  # account % return per trade
        wins = sum(1 for p in pnls_pct if p > 0)
        win_rate = wins / len(trades) * 100

        # Consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls_pct:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Returns
        total_return = (equity[-1] / equity[0] - 1) * 100
        days = len(close)
        annual_return = ((1 + total_return / 100) ** (252 / max(days, 1)) - 1) * 100

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        max_dd = abs(dd.min())

        # Sharpe (annualized, assuming 0 risk-free rate)
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0

        return {
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'annual_return': round(annual_return, 1),
            'max_drawdown': round(max_dd, 1),
            'consecutive_losses': max_consec,
            'total_return': round(total_return, 1),
            'sharpe_ratio': round(sharpe, 2),
            'equity_curve': equity,
            'trades': trades,
        }
