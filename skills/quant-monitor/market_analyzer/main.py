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
            os.makedirs(cfg.output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(equity, color='steelblue')
            ax.axhline(y=args.capital, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"{symbol} Equity Curve")
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
    """Analyze buy-point timing with ML-learned weights and adaptive risk."""
    from entry_detector import EntryDetector
    symbols = args.symbols or cfg.symbols

    for symbol in symbols:
        print(f"\n{'='*70}")
        name = cfg.symbol_names.get(symbol, symbol)
        print(f"  {name} ({symbol}) — 买点分析 (ML驱动)")
        print(f"{'='*70}")

        detector = EntryDetector()
        r = detector.analyze(symbol)

        # Score bar
        bar = '█' * (int(r.entry_score) // 5) + '░' * (20 - int(r.entry_score) // 5)
        print(f"  行情阶段: {r.regime}")
        print(f"  当前价格: {r.entry_price}")
        print(f"  买点评分: {r.entry_score:.0f}/100  [{bar}]")
        print(f"  建议: {r.recommendation}")
        print(f"  {'─'*50}")

        # Stop & Position
        print(f"  止损: {r.stop_loss} (基于: {r.stop_reason})")
        print(f"  风险: -{r.risk_pct:.1f}%  盈亏比: {r.reward_risk}")
        print(f"  推荐仓位: {r.position_pct:.0f}%")
        print(f"  {'─'*50}")

        # Dynamic supports
        print(f"  动态支撑位:")
        for s in r.signals.get('supports', [])[:4]:
            print(f"    {s['price']:.3f}  [{s['type']}] 强度:{s['strength']:.0%}")

        # MA cluster
        ma = r.signals.get('ma_cluster', {})
        if ma:
            below = ma.get('below', {})
            above = ma.get('above', {})
            print(f"  均线: {' | '.join(f'{k}={v:.3f}' for k,v in {**below, **above}.items())}")
            print(f"  价格下方均线: {len(below)}条")

        # Volume + K-line
        vol = r.signals.get('volume', {})
        kline = r.signals.get('kline', {})
        vol_desc = '缩量' if vol.get('shrinking') else ('放量' if vol.get('expanding') else '正常')
        print(f"  量能: {vol_desc} (量比:{vol.get('vol_ratio', 'N/A')})")
        kline_ok = [k for k, v in kline.items() if v and not k.startswith('_')]
        if kline_ok:
            print(f"  K线: {', '.join(kline_ok)}")
        else:
            print(f"  K线: 无反转形态")

        print(f"  {'─'*50}")

        # Historical stats
        if r.similar_count >= 5:
            print(f"  历史统计 (样本{r.similar_count}):")
            print(f"    上涨概率: {r.win_rate:.0f}%  平均涨幅: {r.avg_return:+.1f}%")
            print(f"    平均回撤: -{r.avg_max_dd:.1f}%  90%分位: -{r.dd_90pct:.1f}%  最差: -{r.worst_dd:.1f}%")
        else:
            print(f"  历史统计: 样本不足")

        # Feature importance
        if r.top_features:
            print(f"  因子贡献 (SHAP):")
            for f in r.top_features[:5]:
                print(f"    {f['feature']}: {f['contribution']:.1f}%")

    # Run backtest if requested
    if getattr(args, 'with_backtest', False):
        # Ensure backtest args have defaults from entry's context
        for attr, default in [('capital', 60000), ('position', 20),
                               ('stop_loss', 3), ('take_profit', 5),
                               ('entry_threshold', 70), ('no_charts', False)]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        cmd_backtest(args)

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
        epilog="示例: python main.py entry --symbol 159915 --with-backtest")
    e.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部)")
    e.add_argument("--with-backtest", action="store_true",
                   help="同时运行交易模拟回测")
    e.add_argument("--capital", type=float, default=60000,
                   help="回测本金 (默认: 60000)")
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
