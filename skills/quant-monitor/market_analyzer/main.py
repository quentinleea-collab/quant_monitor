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
    """Train XGBoost models for specified symbols."""
    symbols = args.symbols or cfg.symbols
    engineer = FeatureEngineer(cfg)
    labeler = LabelGenerator(cfg)
    trainer = ModelTrainer(cfg)
    shap_analyzer = SHAPAnalyzer()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  Training: {cfg.symbol_names.get(symbol, symbol)} ({symbol})")
        print(f"{'='*60}")

        # 1. Build features
        print(f"[1/5] Building features...")
        data = engineer.build_features(symbol, cfg.start_date, cfg.end_date)
        close = data["close"]
        features = data.drop(columns=["close"])
        print(f"      {len(features)} samples, {features.shape[1]} features")

        # 2. Generate labels (needs close prices)
        print(f"[2/5] Generating labels (horizon={cfg.label_horizon}d, threshold={cfg.label_threshold:.0%})...")
        labels = labeler.generate(pd.DataFrame({"close": close}))
        stats = labeler.get_label_stats(labels)
        print(f"      Positive: {stats['positive']} ({stats['positive_rate']:.1%}), Negative: {stats['negative']}")

        # 3. Train model (without close to prevent leakage)
        print(f"[3/5] Training XGBoost...")
        model = trainer.train(features, labels, symbol)
        eval_results = trainer.evaluate(model, features, labels)
        print(f"      Accuracy: {eval_results['overall']['accuracy']:.3f}, F1: {eval_results['overall']['f1']:.3f}")
        for t in cfg.score_thresholds:
            key = f'thresh_{t}'
            if key in eval_results:
                r = eval_results[key]
                print(f"      Threshold {t}: {r['signals']} signals, win_rate={r['win_rate']:.1%}")

        # 4. SHAP analysis
        print(f"[4/5] Computing SHAP...")
        valid = labels.notna()
        shap_result = shap_analyzer.analyze(model, features[valid].iloc[-500:])  # last 500 for speed
        print(f"      Top 5 features:")
        for f in shap_result['feature_importance'][:5]:
            print(f"        {f['rank']}. {f['feature']}: {f['importance_pct']}%")

        # 5. Pattern mining
        print(f"[5/5] Mining patterns...")
        miner = PatternMiner(cfg)
        patterns = miner.mine(features[valid], labels[valid])
        print(f"      Top 3 patterns:")
        for p in patterns[:3]:
            names = ', '.join(p['features'][:3])
            print(f"        [{names}]: win={p['win_rate']:.1%}, n={p['sample_count']}")

        # Save
        trainer.save(symbol)

    print(f"\n{'='*60}")
    print(f"  All models trained and saved to {cfg.model_dir}/")
    print(f"{'='*60}")


def cmd_scan(args):
    """Scan all symbols for bottom probability."""
    scanner = DailyScanner(cfg)
    symbols = args.symbols or cfg.symbols
    results = scanner.scan(symbols)

    print()
    print("=" * 80)
    print(f"  市场底部扫描 (中期反弹概率) — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)
    for _, row in results.iterrows():
        prob = row['bottom_prob']
        prob_str = f"{prob:.0f}%" if prob is not None else "N/A"
        name = str(row['name'])
        trend = str(row['trend_state'])
        rec = str(row['recommendation'])
        sl = str(row.get('stop_loss', 'N/A'))
        dd = str(row.get('hist_max_dd', 'N/A'))
        print(f"  {name}")
        print(f"    中期反弹概率: {prob_str}   趋势: {trend}")
        print(f"    建议: {rec}              止损位: {sl}")
        print(f"    相似市况下历史最大跌幅: {dd}")
        print()

    print("=" * 80)
    print("  中期反弹概率 = XGBoost预测未来10个交易日最高涨幅>=3%的概率")
    print("  历史最大跌幅  = 历史上MA60偏离度相近时, 未来20天最大进一步跌幅")
    print("  止损位       = min(20日低点, MA60) × 0.98")
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

    args = p.parse_args()

    if not args.command:
        p.print_help()
        return

    setup_logging(getattr(args, 'verbose', False))

    # Apply CLI date overrides
    if hasattr(args, 'start') and args.start:
        cfg.start_date = args.start
    if hasattr(args, 'end') and args.end:
        cfg.end_date = args.end

    {"train": cmd_train, "scan": cmd_scan, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
