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
import io

# ── Console UTF-8 safety ──────────────────────────────────────
if sys.stdout.encoding is None or sys.stdout.encoding.upper() not in (
    "UTF-8", "UTF8"
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
        label = f"{row['support_type']}({row.get('timeframe', '')})"
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


def _ensure_ma_columns(df: pd.DataFrame, periods, source_col: str = "close") -> pd.DataFrame:
    """Compute MA columns for given periods on a DataFrame if not already present."""
    df = df.copy()
    for p in periods:
        col = f"MA{p}"
        if col not in df.columns:
            df[col] = df[source_col].rolling(p, min_periods=p).mean()
    return df


def _merge_timeframe_ma(
    daily_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    periods,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge timeframe-specific MA values into the daily DataFrame with a column prefix.

    Each MA{period} column from tf_df is merged as {prefix}MA{period} into daily_df
    and forward-filled so the value is available for every daily bar.
    """
    result = daily_df.copy()
    if "date" not in result.columns:
        return result
    tf_df = tf_df.copy()
    tf_df["date"] = pd.to_datetime(tf_df["date"])

    for p in periods:
        src_col = f"MA{p}"
        dst_col = f"{prefix}MA{p}"
        if src_col not in tf_df.columns:
            continue
        temp = tf_df[["date", src_col]].rename(columns={src_col: dst_col})
        result = result.merge(temp, on="date", how="left")
        result[dst_col] = result[dst_col].ffill()
    return result


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # ── Config ────────────────────────────────────────────────
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

    # ── 1. Load data (all timeframes) ────────────────────────
    loader = DataLoader()
    data = loader.load_all(cfg.symbol, cfg.start_date, cfg.end_date)
    df_daily = data["daily"]
    logger.info(f"Loaded {len(df_daily)} daily bars")

    # ── 2. Compute indicators on daily data ──────────────────
    # This adds MA{period} columns for cfg.daily_ma_periods, plus
    # BOLL, ATR, RSI, volume MAs, K-line patterns, etc.
    df = compute_indicators(df_daily, cfg)
    df["date"] = pd.to_datetime(df["date"])

    # ── 3. Multi-timeframe MA computation ────────────────────
    # Compute MAs on their native timeframes, then map back to
    # daily bars via merge + forward-fill.

    # 3a. Weekly MAs (e.g. 10-week MA = 50-day equivalent)
    weekly_df = data["weekly"].copy()
    weekly_df = _ensure_ma_columns(weekly_df, cfg.weekly_ma_periods)
    weekly_df["date"] = pd.to_datetime(weekly_df["date"])

    # 3b. Monthly MAs (e.g. 5-month MA = 100-day equivalent)
    monthly_df = data["monthly"].copy()
    monthly_df = _ensure_ma_columns(monthly_df, cfg.monthly_ma_periods)
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])

    # 3c. Intraday MAs (60min / 120min)
    intraday_dfs = {}
    if "min60" in data:
        min60_df = data["min60"].copy()
        min60_df = _ensure_ma_columns(min60_df, cfg.min60_ma_periods)
        min60_df["date"] = pd.to_datetime(min60_df["date"])
        intraday_dfs["60分钟"] = min60_df
        logger.info(f"60min data: {len(min60_df)} bars")
    if "min120" in data:
        min120_df = data["min120"].copy()
        min120_df = _ensure_ma_columns(min120_df, cfg.min120_ma_periods)
        min120_df["date"] = pd.to_datetime(min120_df["date"])
        intraday_dfs["120分钟"] = min120_df
        logger.info(f"120min data: {len(min120_df)} bars")

    # ── 4. Build daily_with_wm: daily + weekly/monthly MAs ───
    # Weekly MAs get W_MA{p} prefix; monthly MAs get M_MA{p} prefix.
    # This avoids column-name collisions with daily MA{p} columns.
    daily_with_wm = df.copy()
    daily_with_wm = _merge_timeframe_ma(daily_with_wm, weekly_df, cfg.weekly_ma_periods, "W_")
    daily_with_wm = _merge_timeframe_ma(daily_with_wm, monthly_df, cfg.monthly_ma_periods, "M_")

    # ── 5. Detect support levels ──────────────────────────────
    # 5a. Daily detectors (run on df with correct daily MAs)
    detectors_ma = [
        MASupport(list(cfg.daily_ma_periods), "日"),
    ]
    detectors_price = [
        BollingerSupport(),
        PriorLowSupport(cfg.prior_low_windows),
        BoxRangeSupport(cfg),
        TrendLineSupport(cfg),
        VolumeClusterSupport(cfg),
        RoundNumberSupport(),
    ]
    all_daily_detectors = detectors_ma + detectors_price
    supports = combine_detections(all_daily_detectors, df)
    logger.info(
        f"Daily detectors: {len(supports)} rows, "
        f"{supports['support_type'].nunique() if not supports.empty else 0} types"
    )

    # 5b. Weekly MA detection
    # Use a temp copy of daily_with_wm where W_MA{period} columns are
    # mapped to MA{period} names so MASupport can find them.
    if cfg.weekly_ma_periods:
        wk_det = MASupport(list(cfg.weekly_ma_periods), "周")
        wk_df_tmp = daily_with_wm.copy()
        for p in cfg.weekly_ma_periods:
            w_col = f"W_MA{p}"
            if w_col in wk_df_tmp.columns:
                wk_df_tmp[f"MA{p}"] = wk_df_tmp[w_col]
        wk_result = wk_det.detect(wk_df_tmp)
        if not wk_result.empty:
            supports = pd.concat([supports, wk_result], ignore_index=True)
            logger.info(f"Weekly MA supports: {len(wk_result)} rows")

    # 5c. Monthly MA detection
    if cfg.monthly_ma_periods:
        mo_det = MASupport(list(cfg.monthly_ma_periods), "月")
        mo_df_tmp = daily_with_wm.copy()
        for p in cfg.monthly_ma_periods:
            m_col = f"M_MA{p}"
            if m_col in mo_df_tmp.columns:
                mo_df_tmp[f"MA{p}"] = mo_df_tmp[m_col]
        mo_result = mo_det.detect(mo_df_tmp)
        if not mo_result.empty:
            supports = pd.concat([supports, mo_result], ignore_index=True)
            logger.info(f"Monthly MA supports: {len(mo_result)} rows")

    # 5d. Intraday MA detection (detected on native timeframe frames)
    for tf_label, tf_df in intraday_dfs.items():
        periods_map = {
            "60分钟": cfg.min60_ma_periods,
            "120分钟": cfg.min120_ma_periods,
        }
        periods = periods_map.get(tf_label)
        if periods:
            det = MASupport(list(periods), tf_label)
            r = det.detect(tf_df)
            if not r.empty:
                supports = pd.concat([supports, r], ignore_index=True)
                logger.info(f"{tf_label} MA supports: {len(r)} rows")

    # Sort by date and type
    if not supports.empty:
        supports = supports.sort_values(["date", "support_type"]).reset_index(drop=True)

    logger.info(
        f"Detected {len(supports)} support level rows "
        f"({supports['support_type'].nunique() if not supports.empty else 0} types)"
    )

    if supports.empty:
        print("No supports detected. Try a different symbol or date range.")
        return

    # ── 6. Backtest ──────────────────────────────────────────
    # Run on daily_with_wm so that weekly/monthly MA values are
    # available for touch detection on each daily bar.
    engine = BacktestEngine(cfg)
    events = engine.run(daily_with_wm, supports)
    logger.info(f"Backtest: {len(events)} touch events")

    if not events:
        print("No touch events found. Check support data.")
        return

    # ── 7. Confirmation analysis ──────────────────────────────
    conf_analyzer = ConfirmationAnalyzer(cfg)
    events = conf_analyzer.analyze(events, daily_with_wm, supports)

    # ── 8. Statistics ────────────────────────────────────────
    stat_analyzer = StatisticsAnalyzer(cfg)
    stats = stat_analyzer.analyze(events)

    # ── 9. Output ────────────────────────────────────────────
    print_header(stats, cfg.symbol)

    if cfg.generate_excel:
        excel_path = ExcelExporter(cfg).export(stats)
        print(f"Excel 报告: {excel_path}")

    if cfg.generate_charts:
        chart_paths = ChartGenerator(cfg).generate_all(stats, daily_with_wm, supports)
        for p in chart_paths:
            if p:
                print(f"图表: {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
