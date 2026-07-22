#!/usr/bin/env python3
"""
Market Bottom Detector — CLI Entry Point
========================================
XGBoost + SHAP pipeline for A-share bottom probability detection.

Usage:
  python main.py train              # Train models for all symbols
  python main.py scan               # Daily scan (requires trained models)
  python main.py train --symbol 588170  # Train single symbol
"""
import sys, os, logging, argparse, io
import pandas as pd
from datetime import datetime

# Fix Unicode output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ma_config import config as cfg
from feature_engineer import FeatureEngineer
from label_generator import LabelGenerator
from model_trainer import ModelTrainer
from shap_analyzer import SHAPAnalyzer
from pattern_miner import PatternMiner
from daily_scanner import DailyScanner


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def cmd_train(args):
    """Train multi-label XGBoost models for all symbols."""
    symbols = args.symbols or cfg.symbols
    engineer = FeatureEngineer(cfg)
    labeler = LabelGenerator(cfg)
    trainer = ModelTrainer(cfg)

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {cfg.symbol_names.get(symbol, symbol)} ({symbol})")
        print(f"{'='*60}")

        # Build features
        data = engineer.build_features(symbol, cfg.start_date, cfg.end_date)
        close = data["close"]
        features = data.drop(columns=["close"])
        print(f"  Features: {len(features)} samples x {features.shape[1]} dims")

        # Generate multi-labels
        labels_df = labeler.generate_all(pd.DataFrame({"close": close}))
        stats = labeler.get_label_stats(labels_df)
        for col, s in stats.items():
            print(f"  {col}: {s['positive_rate']}% positive ({s['positive']}/{s['total']})")

        # Train all models
        for lt in ['rebound_3pct', 'tp_win', 'sl_loss', 'final_profit']:
            y = labels_df[lt]
            m = trainer.train(features, y, symbol, lt)
            trainer.save(symbol, lt)
            ev = trainer.evaluate(m, features, y)
            print(f"  {lt}: acc={ev['overall']['accuracy']} f1={ev['overall']['f1']} "
                  f"thresh80={ev.get('thresh_80',{}).get('win_rate','N/A')}")

    print(f"\n  Models saved to {cfg.model_dir}/")


def cmd_backtest(args):
    """Run trading simulator on historical predictions."""
    from trading_simulator import TradingSimulator
    import numpy as np

    symbols = args.symbols or cfg.symbols
    engineer = FeatureEngineer(cfg)
    trainer = ModelTrainer(cfg)

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  Backtest: {cfg.symbol_names.get(symbol, symbol)} ({symbol})")
        print(f"{'='*60}")

        data = engineer.build_features(symbol, cfg.start_date, cfg.end_date)
        close = data["close"]
        features = data.drop(columns=["close"])

        # Load model and get predictions
        try:
            trainer.load(symbol, 'rebound_3pct')
        except FileNotFoundError:
            logger.warning("No model — run 'train' first")
            continue

        proba = trainer.predict_proba(features, symbol, 'rebound_3pct') * 100

        sim = TradingSimulator(
            capital=args.capital, position_pct=args.position / 100,
            stop_loss=-args.stop_loss / 100, take_profit=args.take_profit / 100,
        )
        result = sim.run(close, pd.Series(proba, index=features.index), entry_threshold=args.entry_threshold)

        print(f"  总交易: {result['total_trades']}次")
        print(f"  胜率: {result['win_rate']}%")
        print(f"  年化收益: {result['annual_return']}%")
        print(f"  最大回撤: {result['max_drawdown']}%")
        print(f"  最大连续亏损: {result['consecutive_losses']}次")
        print(f"  总收益: {result['total_return']}%")
        print(f"  Sharpe: {result['sharpe_ratio']}")

        # Equity curve chart
        equity = result['equity_curve']
        if len(equity) > 0 and not args.no_charts:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(equity, color='steelblue')
            ax.axhline(y=args.capital, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"{cfg.symbol_names.get(symbol, symbol)} Equity Curve")
            ax.set_ylabel("Account Value")
            path = f"{cfg.output_dir}/{symbol}_equity_{datetime.now().strftime('%Y%m%d')}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  资金曲线: {path}")


def cmd_scan(args):
    """Scan all symbols for bottom probability with multi-outcome analysis."""
    scanner = DailyScanner(cfg)
    symbols = args.symbols or cfg.symbols
    results = scanner.scan(symbols)

    print()
    print("=" * 80)
    print(f"  市场底部扫描 — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)

    for _, row in results.iterrows():
        name = str(row['name'])
        rebound = row['bottom_prob']
        tp_win = row.get('tp_win_prob')
        sl_loss = row.get('sl_loss_prob')
        final_p = row.get('final_profit_prob')
        shap = row.get('shap_signals', [])

        print(f"  {name}")
        print(f"    反弹>=3%概率: {rebound:.0f}%  |  先达+5%概率: {tp_win:.0f}%  |  先触-3%概率: {sl_loss:.0f}%  |  最终盈利概率: {final_p:.0f}%")
        print(f"    趋势: {row['trend_state']}  |  建议: {row['recommendation']}")
        print(f"    止损位: {row.get('stop_loss', 'N/A')}")
        print(f"    历史最差进一步跌幅: {row.get('hist_max_dd', 'N/A')}")

        # SHAP per-signal contributions
        if shap and len(shap) > 0:
            print(f"    当前信号贡献:")
            total_pos = sum(s['contribution'] for s in shap if s['contribution'] > 0)
            total_neg = sum(abs(s['contribution']) for s in shap if s['contribution'] < 0)
            positives = [s for s in shap if s['contribution'] > 0][:5]
            negatives = [s for s in shap if s['contribution'] < 0][:3]
            for s in positives:
                pct = s['contribution'] / total_pos * 100 if total_pos > 0 else 0
                print(f"      +{s['feature']}: +{pct:.0f}%")
            for s in negatives:
                pct = abs(s['contribution']) / total_neg * 100 if total_neg > 0 else 0
                print(f"      -{s['feature']}: -{pct:.0f}%")

        # Historical similarity (top 3)
        sims = row.get('similar_periods', [])
        if sims and len(sims) >= 3:
            print(f"    历史相似形态 (Top 3):")
            for s in sims[:3]:
                sim_pct = s['similarity']
                fwd = f"3日:{s['fwd_3d']:+.1f}%  5日:{s['fwd_5d']:+.1f}%  10日:{s['fwd_10d']:+.1f}%" if s['fwd_3d'] is not None else "N/A"
                print(f"      {s['start_date']}~{s['end_date']}  相似度:{sim_pct:.0f}%  →  {fwd}")
        print()

    print("=" * 80)
    print("  反弹>=3%概率 = XGBoost预测未来10天最高涨幅>=3%的统计概率")
    print("  先达+5%概率  = 未来20天内先触及+5%止盈(而非-3%止损)的概率")
    print("  先触-3%概率  = 未来20天内先触及-3%止损(而非+5%止盈)的概率")
    print("  信号贡献     = SHAP值: 每个信号对今日预测的正/负影响")
    print("=" * 80)
    print()


def cmd_report(args):
    """Full report: train + scan."""
    cmd_train(args)
    cmd_scan(args)


def main():
    p = argparse.ArgumentParser(description="Market Bottom Detector")
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Train models")
    t.add_argument("--symbols", nargs="*", default=None, help="Symbols to train (default: all)")
    t.add_argument("--start", default=cfg.start_date)
    t.add_argument("--end", default=cfg.end_date)
    t.add_argument("--verbose", "-v", action="store_true")

    s = sub.add_parser("scan", help="Daily scan")
    s.add_argument("--symbols", nargs="*", default=None)
    s.add_argument("--verbose", "-v", action="store_true")

    r = sub.add_parser("report", help="Full report (train + scan)")
    r.add_argument("--symbols", nargs="*", default=None)
    r.add_argument("--verbose", "-v", action="store_true")

    b = sub.add_parser("backtest", help="Run trading simulator")
    b.add_argument("--symbols", nargs="*", default=None)
    b.add_argument("--capital", type=float, default=60000, help="Initial capital")
    b.add_argument("--position", type=float, default=20, help="Position size pct")
    b.add_argument("--stop_loss", type=float, default=3, help="Stop loss pct")
    b.add_argument("--take_profit", type=float, default=5, help="Take profit pct")
    b.add_argument("--entry_threshold", type=float, default=70, help="Entry threshold")
    b.add_argument("--no-charts", action="store_true")
    b.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args()

    if not args.command:
        p.print_help()
        return

    setup_logging(getattr(args, 'verbose', False))

    if hasattr(args, 'start') and args.start:
        cfg.start_date = args.start
    if hasattr(args, 'end') and args.end:
        cfg.end_date = args.end

    {"train": cmd_train, "scan": cmd_scan, "report": cmd_report, "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
