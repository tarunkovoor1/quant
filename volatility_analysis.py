#!/usr/local/bin/python3

"""
Volatility Analysis

Measures and visualizes market volatility for daily bar data under
/home/ninan/analytics/data/dbar/{stocks,etfs}/{subdir}/.

Computes:
  - Daily percentage returns and log returns
  - Historical / realized volatility: annualized standard deviation of log
    returns over a configurable rolling lookback window
  - Rolling volatility over fixed 10/20/50-day windows
  - Average True Range (ATR), Wilder-smoothed, configurable period

Displays, for the lookback-based historical volatility series (and again
for each of the 10/20/50-day windows and for ATR): current value, average,
highest period (date + value), lowest period (date + value), and the
full-sample annualized volatility.

Default mode: ALL TICKERS. With no arguments, runs every ticker file found
under both data/dbar/stocks/{ab,cd,...,yz}/ and data/dbar/etfs/{ab,cd,...,yz}/,
writing one aggregated summary CSV per {type, subdir} pair:
    output/volatility/{type}_{subdir}_volatility_summary.csv
(one row-group per ticker). Restrict with --type and/or --subdir. This
matches the --type/--subdir shape the rest of this repo's MAP scripts use,
so it also runs unmodified via run_all.sh, e.g.:
    ./run_all.sh volatility_analysis.py first
    ./run_all.sh volatility_analysis.py second
Per-ticker PNG charts are skipped by default in this mode (tens of thousands
of tickers would mean tens of thousands of charts) — pass --plot to generate
them into output/volatility/charts/.

Single-ticker mode: pass --ticker to analyze just one symbol, print the full
report, and write one summary CSV + one PNG chart.

Usage:
    ./volatility_analysis.py [--type stocks|etfs] [--subdir SUBDIR [SUBDIR ...]]
                              [--lookback N] [--atr-period N]
                              [--trading-days-per-year N] [--plot]

    ./volatility_analysis.py --ticker TICKER [--type stocks|etfs] [--dir DIR]
                              [--lookback N] [--atr-period N]
                              [--trading-days-per-year N]
                              [--summary-out PATH] [--series-out PATH]
                              [--plot-out PATH] [--no-plot]
"""

import argparse
import csv
import glob
import math
import os
from datetime import datetime

import numpy as np

DEFAULT_DBAR_DIR = "/home/ninan/analytics/data/dbar"
DEFAULT_TICKER = "SPY"
DEFAULT_SINGLE_TYPE = "etfs"  # used only when --ticker is given without --type
DEFAULT_LOOKBACK = 20
DEFAULT_ATR_PERIOD = 14
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_OUTPUT_DIR = "/home/ninan/analytics/output/volatility"
ROLLING_WINDOWS = [10, 20, 50]
SUBDIR_PAIRS = ["ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr", "st", "uv", "wx", "yz"]
ASSET_TYPES = ["stocks", "etfs"]


def subdir_for_ticker(ticker):
    """Map a ticker's first letter to this repo's ab/cd/ef/... subdirectory convention."""
    first_letter = ticker[0].lower()
    for pair in SUBDIR_PAIRS:
        if pair[0] <= first_letter <= pair[1]:
            return pair
    return SUBDIR_PAIRS[-1]


def list_ticker_files(directory):
    """All daily-bar ticker files in a subdir, for batch mode."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and "_1day_" in f
    )


def find_ticker_file(directory, ticker):
    patterns = [f"{ticker}_*1day*.txt", f"{ticker}_*.txt"]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No data file found for {ticker} in {directory}")


def load_daily_bars(filepath):
    """
    Parse a firstratedata-style daily bar file: date[,time],open,high,low,close,volume
    Returns a list of dicts sorted chronologically.
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            date_str = parts[0].split(" ")[0]  # strip time-of-day if present
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                open_, high, low, close = (float(parts[1]), float(parts[2]),
                                            float(parts[3]), float(parts[4]))
                volume = float(parts[5]) if len(parts) > 5 else None
            except (ValueError, IndexError):
                continue
            # Bad ticks (zero/negative prices) show up occasionally in the raw
            # data; a real price can never be <= 0, so drop the row rather
            # than let it poison log-return/ATR calculations downstream.
            if open_ <= 0 or high <= 0 or low <= 0 or close <= 0:
                continue
            rows.append({"date": date, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})
    rows.sort(key=lambda r: r["date"])
    return rows


def compute_returns(bars):
    """
    Returns (pct_returns, log_returns), each a list aligned index-for-index
    with `bars` (None at index 0, since there's no prior day to compare to).
    """
    n = len(bars)
    pct_returns = [None] * n
    log_returns = [None] * n
    for i in range(1, n):
        prev_close = bars[i - 1]["close"]
        close = bars[i]["close"]
        if prev_close <= 0 or close <= 0:
            continue  # defensive: load_daily_bars already filters non-positive prices
        pct_returns[i] = (close - prev_close) / prev_close * 100
        log_returns[i] = math.log(close / prev_close)
    return pct_returns, log_returns


def rolling_annualized_vol(log_returns, window, trading_days_per_year):
    """
    Annualized historical/realized volatility (%): rolling sample std-dev of
    log returns over `window` days, annualized by sqrt(trading_days_per_year).
    Returns a list aligned with log_returns (None until enough data exists).
    """
    n = len(log_returns)
    vol = [None] * n
    for i in range(n):
        if log_returns[i] is None:
            continue
        window_start = i - window + 1
        if window_start < 1:  # index 0 is always None (no return there)
            continue
        window_vals = log_returns[window_start:i + 1]
        if any(v is None for v in window_vals):
            continue
        vol[i] = float(np.std(window_vals, ddof=1) * math.sqrt(trading_days_per_year) * 100)
    return vol


def full_sample_annualized_vol(log_returns, trading_days_per_year):
    """Single headline annualized volatility (%) computed over the entire history."""
    vals = [v for v in log_returns if v is not None]
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=1) * math.sqrt(trading_days_per_year) * 100)


def compute_atr(bars, period):
    """
    Wilder-smoothed Average True Range, aligned index-for-index with `bars`
    (None until the first `period` true-range values are available).
    """
    n = len(bars)
    true_ranges = [None] * n
    for i in range(1, n):
        high, low = bars[i]["high"], bars[i]["low"]
        prev_close = bars[i - 1]["close"]
        true_ranges[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))

    atr = [None] * n
    if n <= period:
        return atr

    first_tr_window = true_ranges[1:period + 1]
    if any(v is None for v in first_tr_window):
        return atr

    atr[period] = sum(first_tr_window) / period
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + true_ranges[i]) / period
    return atr


def summarize_series(bars, values, is_percent=True):
    """
    Compute current/average/highest/lowest for a series aligned with `bars`.
    Returns None if there isn't at least one non-None value.
    """
    dated_values = [(bars[i]["date"], v) for i, v in enumerate(values) if v is not None]
    if not dated_values:
        return None
    dates = [d for d, _ in dated_values]
    vals = [v for _, v in dated_values]
    highest_idx = int(np.argmax(vals))
    lowest_idx = int(np.argmin(vals))
    return {
        "current_date": dates[-1],
        "current": vals[-1],
        "average": float(np.mean(vals)),
        "highest_date": dates[highest_idx],
        "highest": vals[highest_idx],
        "lowest_date": dates[lowest_idx],
        "lowest": vals[lowest_idx],
        "is_percent": is_percent,
    }


def fmt(value, is_percent):
    if value is None:
        return "n/a"
    return f"{value:.2f}%" if is_percent else f"{value:.4f}"


def format_series_summary(label, stats):
    lines = [f"\n{label}"]
    if stats is None:
        lines.append("  Not enough data to compute")
        return "\n".join(lines)
    ip = stats["is_percent"]
    lines.append(f"  Current  ({stats['current_date']}) : {fmt(stats['current'], ip)}")
    lines.append(f"  Average                     : {fmt(stats['average'], ip)}")
    lines.append(f"  Highest  ({stats['highest_date']}) : {fmt(stats['highest'], ip)}")
    lines.append(f"  Lowest   ({stats['lowest_date']}) : {fmt(stats['lowest'], ip)}")
    return "\n".join(lines)


def print_series_summary(label, stats):
    print(format_series_summary(label, stats))


def format_ticker_report(ticker, r, lookback, atr_period):
    """Build the same human-readable report the console prints, as a string."""
    lines = [f"{'=' * 70}", f"{ticker} — VOLATILITY ANALYSIS", f"{'=' * 70}"]
    lines.append(format_series_summary(
        f"Historical / Realized Volatility (lookback={lookback}d, annualized)",
        r["hist_vol_stats"]))
    lines.append(f"\nAnnualized Volatility (full sample, {len(r['log_returns']) - 1} returns)"
                  f"\n  {fmt(r['full_sample_vol'], True)}")
    for w in ROLLING_WINDOWS:
        lines.append(format_series_summary(f"Rolling Volatility ({w}-day, annualized)",
                                            r["rolling_stats_by_window"][w]))
    lines.append(format_series_summary(f"Average True Range (period={atr_period})", r["atr_stats"]))
    return "\n".join(lines) + "\n"


def write_readable_report(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def build_summary_rows(ticker, hist_vol_stats, lookback, rolling_stats_by_window,
                        atr_stats, atr_period, full_sample_vol):
    rows = []

    def add_row(metric, stats):
        if stats is None:
            rows.append({"ticker": ticker, "metric": metric, "current_date": "",
                         "current": "", "average": "", "highest_date": "",
                         "highest": "", "lowest_date": "", "lowest": ""})
            return
        rows.append({
            "ticker": ticker, "metric": metric,
            "current_date": stats["current_date"], "current": round(stats["current"], 4),
            "average": round(stats["average"], 4),
            "highest_date": stats["highest_date"], "highest": round(stats["highest"], 4),
            "lowest_date": stats["lowest_date"], "lowest": round(stats["lowest"], 4),
        })

    add_row(f"historical_volatility_lookback_{lookback}", hist_vol_stats)
    for window in ROLLING_WINDOWS:
        add_row(f"rolling_volatility_{window}d", rolling_stats_by_window[window])
    add_row(f"atr_{atr_period}", atr_stats)
    rows.append({"ticker": ticker, "metric": "annualized_volatility_full_sample",
                 "current_date": "", "current": round(full_sample_vol, 4) if full_sample_vol else "",
                 "average": "", "highest_date": "", "highest": "", "lowest_date": "", "lowest": ""})
    return rows


def write_summary_csv(rows, path):
    fieldnames = ["ticker", "metric", "current_date", "current", "average",
                  "highest_date", "highest", "lowest_date", "lowest"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_series_csv(bars, pct_returns, log_returns, hist_vol, rolling_vols, atr, path):
    fieldnames = ["date", "close", "pct_return", "log_return", "hist_vol"] + \
        [f"vol_{w}d" for w in ROLLING_WINDOWS] + ["atr"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, bar in enumerate(bars):
            row = {
                "date": bar["date"], "close": bar["close"],
                "pct_return": round(pct_returns[i], 4) if pct_returns[i] is not None else "",
                "log_return": round(log_returns[i], 6) if log_returns[i] is not None else "",
                "hist_vol": round(hist_vol[i], 4) if hist_vol[i] is not None else "",
                "atr": round(atr[i], 4) if atr[i] is not None else "",
            }
            for w in ROLLING_WINDOWS:
                v = rolling_vols[w][i]
                row[f"vol_{w}d"] = round(v, 4) if v is not None else ""
            writer.writerow(row)


def plot_volatility(bars, hist_vol, lookback, rolling_vols, atr, atr_period, ticker, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [b["date"] for b in bars]
    closes = [b["close"] for b in bars]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})

    axes[0].plot(dates, closes, color="black", linewidth=0.8)
    axes[0].set_title(f"{ticker} — Close Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, hist_vol, label=f"Historical Vol ({lookback}d)", linewidth=1.0)
    for w in ROLLING_WINDOWS:
        axes[1].plot(dates, rolling_vols[w], label=f"Rolling Vol ({w}d)", linewidth=0.8, alpha=0.8)
    axes[1].set_title(f"{ticker} — Annualized Volatility")
    axes[1].set_ylabel("Annualized Vol (%)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, atr, color="darkorange", linewidth=0.8)
    axes[2].set_title(f"{ticker} — Average True Range ({atr_period}d)")
    axes[2].set_ylabel("ATR")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compute_ticker_stats(bars, args):
    """Run the full volatility calculation for one ticker's bars. Returns a dict of all series/stats."""
    pct_returns, log_returns = compute_returns(bars)
    hist_vol = rolling_annualized_vol(log_returns, args.lookback, args.trading_days_per_year)
    rolling_vols = {w: rolling_annualized_vol(log_returns, w, args.trading_days_per_year)
                     for w in ROLLING_WINDOWS}
    atr = compute_atr(bars, args.atr_period)
    full_sample_vol = full_sample_annualized_vol(log_returns, args.trading_days_per_year)

    hist_vol_stats = summarize_series(bars, hist_vol, is_percent=True)
    rolling_stats_by_window = {w: summarize_series(bars, rolling_vols[w], is_percent=True)
                                for w in ROLLING_WINDOWS}
    atr_stats = summarize_series(bars, atr, is_percent=False)

    return {
        "pct_returns": pct_returns, "log_returns": log_returns,
        "hist_vol": hist_vol, "rolling_vols": rolling_vols, "atr": atr,
        "full_sample_vol": full_sample_vol,
        "hist_vol_stats": hist_vol_stats,
        "rolling_stats_by_window": rolling_stats_by_window,
        "atr_stats": atr_stats,
    }


def run_single(args, ticker):
    asset_type = args.type or DEFAULT_SINGLE_TYPE
    directory = args.dir or os.path.join(args.dbar_dir, asset_type, subdir_for_ticker(ticker))
    filepath = find_ticker_file(directory, ticker)
    print(f"Loading {filepath} ...")

    bars = load_daily_bars(filepath)
    if len(bars) < max(ROLLING_WINDOWS) + args.atr_period + 5:
        raise SystemExit("Not enough daily bars found to analyze.")
    print(f"Loaded {len(bars)} trading days ({bars[0]['date']} to {bars[-1]['date']})")

    r = compute_ticker_stats(bars, args)

    report_text = format_ticker_report(ticker, r, args.lookback, args.atr_period)
    print(report_text)

    if not args.no_summary_out:
        summary_path = args.summary_out or os.path.join(
            DEFAULT_OUTPUT_DIR, f"{ticker}_volatility_summary.csv")
        rows = build_summary_rows(ticker, r["hist_vol_stats"], args.lookback,
                                   r["rolling_stats_by_window"], r["atr_stats"], args.atr_period,
                                   r["full_sample_vol"])
        write_summary_csv(rows, summary_path)
        print(f"\nWrote volatility summary to {summary_path}")

    if not args.no_readable_out:
        readable_path = args.readable_out or os.path.join(
            DEFAULT_OUTPUT_DIR, f"{ticker}_volatility_readable.txt")
        write_readable_report(report_text, readable_path)
        print(f"Wrote human-readable report to {readable_path}")

    if args.series_out:
        write_series_csv(bars, r["pct_returns"], r["log_returns"], r["hist_vol"],
                          r["rolling_vols"], r["atr"], args.series_out)
        print(f"Wrote full daily series to {args.series_out}")

    if not args.no_plot:
        plot_path = args.plot_out or os.path.join(
            DEFAULT_OUTPUT_DIR, f"{ticker}_volatility.png")
        plot_volatility(bars, r["hist_vol"], args.lookback, r["rolling_vols"], r["atr"],
                         args.atr_period, ticker, plot_path)
        print(f"Wrote volatility chart to {plot_path}")


def run_batch_for_type(args, subdirs, asset_type):
    """Every ticker file for one asset type across the given subdir(s), one summary CSV per subdir."""
    min_required = max(ROLLING_WINDOWS) + args.atr_period + 5
    type_processed = 0
    type_skipped = 0

    for subdir in subdirs:
        directory = os.path.join(args.dbar_dir, asset_type, subdir)
        ticker_files = list_ticker_files(directory)
        print(f"\n[{asset_type}/{subdir}] {len(ticker_files)} ticker file(s) found in {directory}")

        rows = []
        readable_blocks = []
        processed = 0
        skipped = 0

        for fname in ticker_files:
            ticker = fname.split("_full_1day")[0]
            filepath = os.path.join(directory, fname)
            try:
                bars = load_daily_bars(filepath)
                if len(bars) < min_required:
                    skipped += 1
                    continue

                r = compute_ticker_stats(bars, args)
                rows.extend(build_summary_rows(ticker, r["hist_vol_stats"], args.lookback,
                                                r["rolling_stats_by_window"], r["atr_stats"],
                                                args.atr_period, r["full_sample_vol"]))
                if not args.no_readable_out:
                    readable_blocks.append(format_ticker_report(ticker, r, args.lookback, args.atr_period))
                processed += 1

                if args.plot:
                    chart_path = os.path.join(DEFAULT_OUTPUT_DIR, "charts", f"{ticker}_volatility.png")
                    plot_volatility(bars, r["hist_vol"], args.lookback, r["rolling_vols"], r["atr"],
                                     args.atr_period, ticker, chart_path)
            except Exception as e:
                # Never let one bad ticker (corrupt file, unexpected data) abort
                # the whole multi-thousand-ticker batch run.
                print(f"  [{asset_type}/{subdir}] Skipping {ticker}: {e}")
                skipped += 1
                continue

        summary_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{asset_type}_{subdir}_volatility_summary.csv")
        write_summary_csv(rows, summary_path)
        print(f"[{asset_type}/{subdir}] Processed {processed}, skipped {skipped} "
              f"(insufficient history or unreadable). Wrote {summary_path}")

        if not args.no_readable_out:
            readable_path = os.path.join(
                DEFAULT_OUTPUT_DIR, f"{asset_type}_{subdir}_volatility_readable.txt")
            write_readable_report("\n".join(readable_blocks), readable_path)
            print(f"[{asset_type}/{subdir}] Wrote human-readable report to {readable_path}")

        type_processed += processed
        type_skipped += skipped

    print(f"\n[{asset_type}] Subtotal: {type_processed} ticker(s) processed, {type_skipped} skipped, "
          f"across {len(subdirs)} subdir(s).")
    return type_processed, type_skipped


def run_batch(args):
    types = [args.type] if args.type else ASSET_TYPES
    subdirs = SUBDIR_PAIRS if not args.subdir or args.subdir == ["all"] else args.subdir

    grand_processed = 0
    grand_skipped = 0
    for asset_type in types:
        processed, skipped = run_batch_for_type(args, subdirs, asset_type)
        grand_processed += processed
        grand_skipped += skipped

    print(f"\nGrand total: {grand_processed} ticker(s) processed, {grand_skipped} skipped, "
          f"across {len(types)} asset type(s) and {len(subdirs)} subdir(s).")


def main():
    parser = argparse.ArgumentParser(description="Measure and visualize market volatility")
    parser.add_argument("--ticker", help="Analyze just this one ticker (single-ticker mode) "
                                          "instead of every ticker (the default)")
    parser.add_argument("--type", choices=ASSET_TYPES,
                         help="Restrict to 'stocks' or 'etfs' (default: both, in all-tickers mode)")
    parser.add_argument("--subdir", nargs="+",
                         help="One or more ab/cd/.../yz subdirs, or 'all' (default: all subdirs)")
    parser.add_argument("--dbar-dir", default=DEFAULT_DBAR_DIR,
                         help="Root dbar directory containing {stocks,etfs}/{subdir}/ files")
    parser.add_argument("--dir", help="Single-ticker mode only: explicit directory containing the "
                                       "ticker's daily bar file (overrides auto-resolution)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                         help="Rolling window (days) for the historical/realized volatility series")
    parser.add_argument("--atr-period", type=int, default=DEFAULT_ATR_PERIOD,
                         help="Period (days) for the Wilder-smoothed ATR")
    parser.add_argument("--trading-days-per-year", type=int, default=DEFAULT_TRADING_DAYS_PER_YEAR,
                         help="Annualization factor (sqrt of this multiplies daily std-dev)")
    parser.add_argument("--summary-out",
                         help="Single-ticker mode: path for the aggregate stats CSV "
                              f"(default: {DEFAULT_OUTPUT_DIR}/<ticker>_volatility_summary.csv)")
    parser.add_argument("--no-summary-out", action="store_true",
                         help="Single-ticker mode: skip writing the summary CSV")
    parser.add_argument("--readable-out",
                         help="Single-ticker mode: path for the human-readable .txt report "
                              f"(default: {DEFAULT_OUTPUT_DIR}/<ticker>_volatility_readable.txt)")
    parser.add_argument("--no-readable-out", action="store_true",
                         help="Skip writing the human-readable .txt report(s) (single-ticker: one "
                              "file; all-tickers mode: one {type}_{subdir}_volatility_readable.txt "
                              "per subdir, at a fixed path)")
    parser.add_argument("--series-out", help="Single-ticker mode: optional path to dump the full "
                                              "daily time series as CSV")
    parser.add_argument("--plot-out",
                         help="Single-ticker mode: path for the PNG chart "
                              f"(default: {DEFAULT_OUTPUT_DIR}/<ticker>_volatility.png)")
    parser.add_argument("--no-plot", action="store_true",
                         help="Single-ticker mode: skip generating the chart")
    parser.add_argument("--plot", action="store_true",
                         help="All-tickers mode: also generate a per-ticker PNG chart "
                              "(off by default; charts/<ticker>_volatility.png)")
    args = parser.parse_args()

    if args.ticker:
        run_single(args, args.ticker)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
