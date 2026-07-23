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


def cmd_entry(args):
    """Analyze buy-point timing within a bottom zone."""
    from entry_detector import EntryDetector
    symbols = args.symbols or cfg.symbols

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"  {cfg.symbol_names.get(symbol, symbol)} ({symbol}) — 买点分析")
        print(f"{'='*70}")

        # First get bottom probability from scanner
        from daily_scanner import DailyScanner
        try:
            scanner = DailyScanner(cfg)
            results = scanner.scan([symbol])
            bp = results.iloc[0]['bottom_prob'] if len(results) > 0 else None
        except Exception:
            bp = None

        # Run entry detector
        detector = EntryDetector()
        result = detector.analyze(symbol, bottom_prob=bp)

        score = result['entry_score']
        bar = '█' * (score // 5) + '░' * (20 - score // 5)

        print(f"  底部概率: {result['bottom_context']}")
        print(f"  当前价格: {result['entry_price']}")
        print(f"  买点评分: {score}/100  [{bar}]")
        print(f"  建议: {result['recommendation']}")
        print(f"  置信度: {result['confidence']}")
        print(f"  {'─'*50}")
        print(f"  逐级止损:")
        print(f"    初始止损:  {result['stop_initial']}")
        print(f"    保本止损:  {result['stop_breakeven']}")
        print(f"    移动止盈:  {result['stop_trail_tight']}")
        print(f"  {'─'*50}")

        # Volume detail
        vol = result.get('volume_signal', {})
        if vol:
            print(f"  量能: {vol.get('shrinkage','')} {vol.get('expansion','')} "
                  f"{vol.get('pattern','')} (量比:{vol.get('vol_ratio','')})")

        # Intraday detail
        intra = result.get('intraday_signal', {})
        if intra:
            if intra.get('fallback'):
                print(f"  分时: {intra['status']}")
            else:
                print(f"  60分钟: {intra.get('m60_macd','')} RSI={intra.get('m60_rsi','')} {intra.get('m60_rsi_signal','')}")
                print(f"  120分钟: RSI={intra.get('m120_rsi','')} {intra.get('m120_rsi_signal','')}")

        # Pattern detail
        pat = result.get('pattern_signal', {})
        if pat:
            signals = [v for k, v in pat.items() if not k.startswith('_')]
            if signals:
                print(f"  K线形态: {', '.join(signals)}")

        # Support detail
        sup = result.get('support_signal', {})
        if sup:
            print(f"  支撑位: {sup.get('level','')} "
                  f"(最近前低:{sup.get('nearest_support','')} "
                  f"距离:{sup.get('distance_pct','')}% "
                  f"MA60:{sup.get('ma60','')})")

        # Multi-TF
        mtf = result.get('multitf_signal', {})
        if mtf:
            print(f"  多周期: {mtf.get('daily_rsi','')} {mtf.get('daily_macd','')} "
                  f"{mtf.get('tf_alignment','')}")

        # What's holding back the score
        if score < 70:
            missing = []
            vol = result.get('volume_signal', {})
            if vol.get('expansion') == '未放量':
                missing.append('等待放量(量>MA5×1.1)')
            if not intra.get('m60_macd') and not intra.get('fallback'):
                missing.append('等待60分钟MACD拐头')
            elif intra.get('fallback'):
                missing.append('分时数据不可用')
            if not pat:
                missing.append('等待K线反转形态(锤子线/吞没/启明星)')
            sup = result.get('support_signal', {})
            if sup.get('distance_pct', 99) > 5:
                missing.append(f"距支撑位{sup.get('distance_pct','')}%偏远")
            if missing:
                print(f"  等待信号: {' | '.join(missing)}")

    print()


def main():
    epi = (
        "代码规则: 6xxxxx=上交所  0xxxxx/3xxxxx=深交所  1xxxxx=深交所ETF\n"
        "示例: python main.py train  --symbol 600519            # 训练\n"
        "      python main.py scan   --symbol 600519            # 扫描\n"
        "      python main.py entry  --symbol 600519            # 买点分析\n"
        "      python main.py backtest --symbol 600519 --capital 100000"
    )
    p = argparse.ArgumentParser(
        description="Market Bottom Detector — XGBoost+SHAP A股底部概率检测",
        epilog=epi,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Train models for specified stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py train --symbol 600519 000858 300750")
    t.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码, 空格分隔 (默认: 000001 399001 399006 159915)")
    t.add_argument("--start", default=cfg.start_date, help="起始日期 YYYYMMDD")
    t.add_argument("--end", default=cfg.end_date, help="结束日期 YYYYMMDD")
    t.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    s = sub.add_parser("scan", help="Scan current market for bottom signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py scan --symbol 600519")
    s.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部已训练标的)")
    s.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    r = sub.add_parser("report", help="Train + scan in one step")
    r.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部)")
    r.add_argument("--verbose", "-v", action="store_true")

    b = sub.add_parser("backtest", help="Simulate trading with historical signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py backtest --symbol 600519 --capital 100000 --position 20")
    b.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部已训练标的)")
    b.add_argument("--capital", type=float, default=60000,
                   help="初始本金, 默认 60000")
    b.add_argument("--position", type=float, default=20,
                   help="每次建仓仓位%%, 默认 20")
    b.add_argument("--stop_loss", type=float, default=3,
                   help="止损线%%, 默认 3 (即-3%%止损)")
    b.add_argument("--take_profit", type=float, default=5,
                   help="止盈线%%, 默认 5 (即+5%%止盈减半)")
    b.add_argument("--entry_threshold", type=float, default=70,
                   help="底部概率入场阈值, 默认 70")
    b.add_argument("--no-charts", action="store_true",
                   help="不生成资金曲线图")
    b.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    e = sub.add_parser("entry", help="Analyze buy-point timing within bottom zone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py entry --symbol 159915")
    e.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部)")
    e.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = p.parse_args()

    if not args.command:
        p.print_help()
        return

    setup_logging(getattr(args, 'verbose', False))

    if hasattr(args, 'start') and args.start:
        cfg.start_date = args.start
    if hasattr(args, 'end') and args.end:
        cfg.end_date = args.end

    {"train": cmd_train, "scan": cmd_scan, "report": cmd_report, "backtest": cmd_backtest, "entry": cmd_entry}[args.command](args)


if __name__ == "__main__":
    main()
