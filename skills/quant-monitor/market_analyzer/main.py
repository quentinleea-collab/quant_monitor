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
import numpy as np
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

        # Compute ATR(14), aligned to feature dates
        c_arr = close.values
        tr_arr = np.maximum(np.abs(c_arr[1:] - c_arr[:-1]), c_arr[1:] - c_arr[:-1])
        atr_arr = np.zeros(len(c_arr))
        for i in range(14, len(c_arr)):
            atr_arr[i] = np.mean(tr_arr[i-13:i+1])
        atr_arr[:14] = atr_arr[14] if len(c_arr) > 14 else c_arr[0] * 0.03
        # Map to feature index: take ATR values where close dates match feature dates
        atr_vals = pd.Series(atr_arr, index=close.index).reindex(features.index).fillna(atr_arr[-1])
        use_atr = getattr(args, 'stop_loss', 0) <= 0

        sim = TradingSimulator(
            capital=args.capital,
            position_pct=args.position / 100,
            stop_loss=args.stop_loss / 100 if not use_atr else None,
            take_profit=args.take_profit / 100,
            max_hold=getattr(args, 'max_hold', 10),
        )
        result = sim.run(
            close, pd.Series(proba, index=features.index),
            entry_threshold=args.entry_threshold,
            atr_series=atr_vals if use_atr else None,
            atr_multiple=getattr(args, 'atr_multiple', 2.0),
        )

        print(f"  总交易: {result['total_trades']}次")
        print(f"  胜率: {result['win_rate']}%")
        print(f"  平均收益: {result['avg_return']}%/笔")
        print(f"  年化收益: {result['annual_return']}%")
        print(f"  最大回撤: {result['max_drawdown']}%")
        print(f"  最大连续亏损: {result['consecutive_losses']}次")
        print(f"  Sharpe: {result['sharpe_ratio']}")

        # Auto-optimize take-profit if requested
        if getattr(args, 'optimize', False):
            _optimize_take_profit(features, close, proba, symbol, atr_vals, args)

        # Equity curve chart with dates + trade markers
        equity = result['equity_curve']
        if len(equity) > 0 and not args.no_charts:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            os.makedirs(cfg.output_dir, exist_ok=True)

            # Use feature dates for x-axis
            feat_idx = features.index[:len(equity)]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(feat_idx, equity, color='steelblue', linewidth=1.5, label='Equity')
            ax.axhline(y=args.capital, color='gray', linestyle='--', alpha=0.5, label='Initial')

            # Mark entries (green ^) and exits (red v for loss, green v for win)
            for t in result.get('trades', []):
                ei = t.get('entry_date')
                xi = t.get('exit_date')
                if ei is not None and ei < len(feat_idx):
                    ax.scatter(feat_idx[ei], equity[ei], color='blue', marker='^', s=80, zorder=5)
                if xi is not None and xi < len(feat_idx):
                    c = 'red' if t.get('pnl_pct', 0) < 0 else 'lime'
                    ax.scatter(feat_idx[xi], equity[xi], color=c, marker='v', s=80, zorder=5)

            ret_total = (result['equity_curve'][-1] / result['equity_curve'][0] - 1) * 100
            ax.set_title(f"{symbol} ({result['total_trades']} trades, "
                        f"{result['win_rate']}% win, {ret_total:.1f}% return)")
            ax.set_ylabel("Value (CNY)")
            ax.legend(loc='upper left')
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()

            path = f"{cfg.output_dir}/{symbol}_equity_{datetime.now().strftime('%Y%m%d')}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  资金曲线: {path}")


def _optimize_take_profit(features, close, proba, symbol, atr_vals, args):
    """Grid-search optimal take-profit level from historical data."""
    from trading_simulator import TradingSimulator
    tp_levels = [3, 4, 5, 6, 7, 8, 10]
    results = []
    print(f"\n  {'─'*50}")
    print(f"  止盈参数优化 (网格搜索):")
    print(f"  {'TP%':<8} {'交易':<6} {'胜率':<8} {'年化':<8} {'回撤':<8} {'Sharpe':<8} {'评分':<8}")
    print(f"  {'─'*50}")

    best_score = -999
    best_tp = 5
    for tp in tp_levels:
        sim = TradingSimulator(
            capital=args.capital, position_pct=args.position / 100,
            stop_loss=args.stop_loss / 100 if args.stop_loss > 0 else None,
            take_profit=tp / 100,
            max_hold=getattr(args, 'max_hold', 10),
        )
        r = sim.run(close, pd.Series(proba, index=features.index),
                    entry_threshold=args.entry_threshold,
                    atr_series=atr_vals if args.stop_loss <= 0 else None)
        if r['total_trades'] < 3:
            continue
        # Composite score: Sharpe × win_rate / max_drawdown
        score = r['sharpe_ratio'] * r['win_rate'] / max(r['max_drawdown'], 0.1)
        results.append((tp, r, score))
        marker = ' ←' if score > best_score else ''
        if score > best_score:
            best_score = score
            best_tp = tp
        print(f"  {tp}%       {r['total_trades']:<6} {r['win_rate']:<8.1f} {r['annual_return']:<8.1f} "
              f"{r['max_drawdown']:<8.1f} {r['sharpe_ratio']:<8.2f} {score:<8.1f}{marker}")

    print(f"  {'─'*50}")
    print(f"  推荐止盈: {best_tp}% (Sharpe×胜率/回撤 综合最优)")
    print()

    # Also scan stop-loss multipliers
    print(f"  ATR止损倍数优化:")
    print(f"  {'倍数':<8} {'交易':<6} {'胜率':<8} {'年化':<8} {'回撤':<8} {'Sharpe':<8} {'评分':<8}")
    print(f"  {'─'*50}")
    best_sl = 2.0; best_sl_score = -999
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        sim = TradingSimulator(
            capital=args.capital, position_pct=args.position / 100,
            stop_loss=None, take_profit=best_tp / 100,
            max_hold=getattr(args, 'max_hold', 10),
        )
        r = sim.run(close, pd.Series(proba, index=features.index),
                    entry_threshold=args.entry_threshold,
                    atr_series=atr_vals, atr_multiple=mult)
        if r['total_trades'] < 5:
            continue
        score = r['sharpe_ratio'] * r['win_rate'] / max(r['max_drawdown'], 0.1)
        marker = ' ←' if score > best_sl_score else ''
        if score > best_sl_score:
            best_sl_score = score; best_sl = mult
        print(f"  {mult}×       {r['total_trades']:<6} {r['win_rate']:<8.1f} {r['annual_return']:<8.1f} "
              f"{r['max_drawdown']:<8.1f} {r['sharpe_ratio']:<8.2f} {score:<8.1f}{marker}")
    print(f"  {'─'*50}")
    print(f"  推荐: --take_profit {best_tp} --atr_multiple {best_sl}")
    print()


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


def cmd_regime(args):
    """Unified trend + bottom + top prediction."""
    from market_regime import MarketRegime
    symbols = args.symbols or cfg.symbols
    mr = MarketRegime()

    # Ensure trend + top models are trained
    for symbol in symbols:
        for lt in ['trend', 'top']:
            try:
                mr._load(symbol, lt)
            except FileNotFoundError:
                print(f"  Training regime models for {symbol}...")
                mr.train_all(symbol)
                break

    # Note: bottom prob requires separate 'scan' command
    # Regime command shows trend + top only

    # Predict
    for symbol in symbols:
        r = mr.predict_all(symbol)
        name = cfg.symbol_names.get(symbol, symbol)
        current = r.get('current_trend', r.get('trend', '未知'))
        forward = r.get('forward_trend', r.get('trend', '未知'))
        conf = r.get('trend_confidence', 0)
        detail = r.get('trend_detail', {})
        top = r.get('top_prob')
        action = r.get('action', '')

        print(f"\n  {name} ({symbol})")
        print(f"  {'─'*50}")
        print(f"  当前趋势: {current}    未来20日预测: {forward} ({conf:.0f}%)")
        if detail:
            print(f"    预测细节: 下跌{detail.get('下跌',0):.0f}%  横盘{detail.get('横盘',0):.0f}%  上涨{detail.get('上涨',0):.0f}%")
        print(f"  顶部概率: {top:.0f}%" if top is not None else "  顶部概率: 未训练")
        print(f"  建议: {action}")
        print(f"  (底部概率请运行: python main.py scan --symbol {symbol})")
    print()


def cmd_exit(args):
    """Compare 4 exit strategies across 3 market regimes."""
    from exit_backtest import ExitBacktest
    symbols = args.symbols or cfg.symbols

    for symbol in symbols:
        eb = ExitBacktest()
        df = eb.compare(symbol)
        if df.empty:
            print(f"\n  {cfg.symbol_names.get(symbol, symbol)}: 数据不足, 跳过")
            continue

        name = cfg.symbol_names.get(symbol, symbol)
        print(f"\n{'='*85}")
        print(f"  {name} ({symbol}) — 退出策略对比")
        print(f"{'='*85}")

        # Best model per regime (only for regimes with enough data)
        for regime in ['down', 'sideways', 'up']:
            subset = df[df['行情'] == regime].sort_values('Sharpe', ascending=False)
            if subset.empty:
                continue
            best = subset.iloc[0]
            regime_cn = {'down': '下跌', 'sideways': '横盘', 'up': '上涨'}.get(regime, regime)
            note = ''
            if best['交易'] < 5:
                note = ' [!] LOW_SAMPLES'
            elif regime == 'up':
                note = ' (底部信号在上涨市天然稀少)'
            print(f"\n  ▸ {regime_cn}市 (最优: {best['模型']}){note}")
            print(f"    {best['交易']}笔  胜率{best['胜率%']:.0f}%  均收益{best['均收益%']:+.2f}%  "
                  f"盈亏比{best['盈亏比']:.1f}  Sharpe{best['Sharpe']:.2f}  退出:{best['退出方式']}")

        # Full table
        print(f"\n  {'模型':<18} {'行情':<6} {'交易':<5} {'胜率':<7} {'均收益':<8} {'持仓天':<7} {'回撤':<7} {'盈亏比':<7} {'Sharpe':<7}")
        print(f"  {'─'*80}")
        for _, r in df.sort_values(['行情', 'Sharpe'], ascending=[True, False]).iterrows():
            print(f"  {r['模型']:<18} {r['行情']:<6} {r['交易']:<5} {r['胜率%']:<7.1f} "
                  f"{r['均收益%']:<8.2f} {r['均持仓天']:<7.1f} {r['最大回撤%']:<7.1f} "
                  f"{r['盈亏比']:<7.1f} {r['Sharpe']:<7.2f}")

        print()

    print("  ★ 推荐: 底部信号的退出策略重点看下跌市和横盘市")
    print("    上涨市中底部信号天然稀少(无需抄底), 退出策略参考意义有限")
    print(f"    当前API仅返回~500条日线(约2年), 更长历史需换数据源")
    print()


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
    b.add_argument("--stop_loss", type=float, default=0,
                   help="止损%%, 0=ATR自适应(推荐), 或输入固定值如3")
    b.add_argument("--take_profit", type=float, default=5,
                   help="止盈%%, 默认 5")
    b.add_argument("--max_hold", type=int, default=10,
                   help="最大持仓天数, 默认 10")
    b.add_argument("--atr_multiple", type=float, default=2.0,
                   help="ATR止损倍数, 默认 2.0")
    b.add_argument("--entry_threshold", type=float, default=70,
                   help="底部概率入场阈值, 默认 70")
    b.add_argument("--optimize", action="store_true",
                   help="自动优化止盈和止损参数")
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

    x = sub.add_parser("exit", help="Compare exit strategies across market regimes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py exit --symbol 159915")
    x.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部)")
    x.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    g = sub.add_parser("regime", help="Trend + bottom + top unified prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python main.py regime --symbol 159915")
    g.add_argument("--symbol", "--symbols", nargs="*", default=None, dest="symbols",
                   help="股票代码 (默认: 全部)")
    g.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = p.parse_args()

    if not args.command:
        p.print_help()
        return

    setup_logging(getattr(args, 'verbose', False))

    if hasattr(args, 'start') and args.start:
        cfg.start_date = args.start
    if hasattr(args, 'end') and args.end:
        cfg.end_date = args.end

    {"train": cmd_train, "scan": cmd_scan, "report": cmd_report,
     "backtest": cmd_backtest, "entry": cmd_entry, "exit": cmd_exit,
     "regime": cmd_regime}[args.command](args)


if __name__ == "__main__":
    main()
