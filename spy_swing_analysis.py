#!/usr/local/bin/python3

"""
Swing Analysis

Reads daily bar data under /home/ninan/analytics/data/dbar/{stocks,etfs}/{subdir}/
and splits historical price action into up-swings and down-swings using a
percentage zigzag: a swing continues in its current direction as long as
price keeps making new extremes, and ends only once price reverses by at
least --reversal-pct (default 1%) off that extreme. Swings are classified by
direction into percent-change magnitude buckets and summarized (count,
median days held, percentile distribution of days held).

It also runs a separate day-over-day analysis: for each threshold in
--gap-thresholds (default 1% through 10%), it finds every day whose
day-over-day close change exceeds that threshold ("swing days"), measures
the number of trading days between consecutive swing days (separately for
up days, down days, and combined), and reports the percentile distribution
of those gaps.

Default mode: ALL TICKERS. With no arguments, runs every ticker file found
under both data/dbar/stocks/{ab,...,yz}/ and data/dbar/etfs/{ab,...,yz}/,
writing one aggregated CSV pair per {type, subdir}:
    output/visionaggr/{type}_{subdir}_swing_summary.csv
    output/visionaggr/{type}_{subdir}_swing_gap_summary.csv
(each with one row-group per ticker, tagged by a `ticker` column). Restrict
scope with --type and/or --subdir. This matches the --type/--subdir shape
the rest of this repo's MAP scripts use, so it also runs unmodified via
run_all.sh, e.g.:
    ./run_all.sh spy_swing_analysis.py first
    ./run_all.sh spy_swing_analysis.py second

Single-ticker mode: pass --ticker to analyze just one symbol and print the
full report (plus, optionally, --csv-out to dump every individual swing).

Usage:
    ./spy_swing_analysis.py [--type stocks|etfs] [--subdir SUBDIR [SUBDIR ...]]
                            [--reversal-pct PCT] [--gap-thresholds 1,2,...,10]

    ./spy_swing_analysis.py --ticker TICKER [--type stocks|etfs] [--dir DIR]
                            [--reversal-pct PCT] [--csv-out PATH]
                            [--gap-thresholds 1,2,3,...,10]
"""

import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np

DEFAULT_DBAR_DIR = "/home/ninan/analytics/data/dbar"
DEFAULT_TICKER = "SPY"
DEFAULT_SINGLE_TYPE = "etfs"  # used only when --ticker is given without --type
DEFAULT_REVERSAL_PCT = 1.0
DEFAULT_OUTPUT_DIR = "/home/ninan/analytics/output/visionaggr"
SUBDIR_PAIRS = ["ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr", "st", "uv", "wx", "yz"]
ASSET_TYPES = ["stocks", "etfs"]

# (low_pct_inclusive, high_pct_exclusive, label)
BUCKETS = [
    (1, 2, "1% to <2%"),
    (2, 3, "2% to <3%"),
    (3, 4, "3% to <4%"),
    (4, 5, "4% to <5%"),
    (5, float("inf"), ">=5%"),
]

PERCENTILES = [1, 25, 50, 75, 85, 90, 95, 99]
DEFAULT_GAP_THRESHOLDS = list(range(1, 11))  # 1% through 10%
MIN_TRADING_DAYS = 60  # skip tickers with too little history to be meaningful


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


def load_daily_closes(filepath):
    """
    Parse a firstratedata-style daily bar file: date[,time],open,high,low,close,volume
    Returns a list of (date, close) tuples sorted chronologically.
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
                close = float(parts[4])
            except (ValueError, IndexError):
                continue
            # Bad ticks (zero/negative prices) show up occasionally in the raw
            # data; a real price can never be <= 0, so drop the row rather
            # than let it poison pct-change/swing calculations downstream.
            if close <= 0:
                continue
            rows.append((date, close))
    rows.sort(key=lambda r: r[0])
    return rows


def build_swing(direction, start_date, end_date, start_price, end_price, start_idx, end_idx):
    pct_change = (end_price - start_price) / start_price * 100
    return {
        "direction": direction,
        "start_date": start_date,
        "end_date": end_date,
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": pct_change,
        "trading_days": end_idx - start_idx,
    }


def detect_swings(daily_prices, reversal_pct):
    """
    Percentage zigzag swing detection.

    While in an 'up' swing, the running extreme is the highest close seen
    since the swing began; the swing ends (at that extreme) once price
    closes reversal_pct% or more below it. Symmetric logic applies to 'down'
    swings. The new swing then starts exactly at the prior swing's extreme.
    """
    if len(daily_prices) < 2:
        return []

    swings = []

    swing_start_idx = 0
    swing_start_date, swing_start_price = daily_prices[0]

    extreme_idx = 0
    extreme_date, extreme_price = daily_prices[0]
    direction = None  # 'up' or 'down', undetermined until the first real move

    for i in range(1, len(daily_prices)):
        date, price = daily_prices[i]

        if direction is None:
            if price > extreme_price:
                direction = "up"
                extreme_idx, extreme_date, extreme_price = i, date, price
            elif price < extreme_price:
                direction = "down"
                extreme_idx, extreme_date, extreme_price = i, date, price
            continue

        if direction == "up":
            if price >= extreme_price:
                extreme_idx, extreme_date, extreme_price = i, date, price
            else:
                drop_pct = (extreme_price - price) / extreme_price * 100
                if drop_pct >= reversal_pct:
                    swings.append(build_swing(
                        "up", swing_start_date, extreme_date,
                        swing_start_price, extreme_price,
                        swing_start_idx, extreme_idx,
                    ))
                    swing_start_idx, swing_start_date, swing_start_price = (
                        extreme_idx, extreme_date, extreme_price,
                    )
                    direction = "down"
                    extreme_idx, extreme_date, extreme_price = i, date, price
        else:  # direction == "down"
            if price <= extreme_price:
                extreme_idx, extreme_date, extreme_price = i, date, price
            else:
                rise_pct = (price - extreme_price) / extreme_price * 100
                if rise_pct >= reversal_pct:
                    swings.append(build_swing(
                        "down", swing_start_date, extreme_date,
                        swing_start_price, extreme_price,
                        swing_start_idx, extreme_idx,
                    ))
                    swing_start_idx, swing_start_date, swing_start_price = (
                        extreme_idx, extreme_date, extreme_price,
                    )
                    direction = "up"
                    extreme_idx, extreme_date, extreme_price = i, date, price

    # Close out the final, still-open swing using the latest extreme reached.
    if direction is not None and extreme_idx != swing_start_idx:
        swings.append(build_swing(
            direction, swing_start_date, extreme_date,
            swing_start_price, extreme_price,
            swing_start_idx, extreme_idx,
        ))

    return swings


def bucket_for(pct_change):
    abs_pct = abs(pct_change)
    for low, high, label in BUCKETS:
        if low <= abs_pct < high:
            return label
    return None


def summarize(swings):
    """Group swings by direction and magnitude bucket, computing stats on trading days held."""
    grouped = {
        "up": {label: [] for _, _, label in BUCKETS},
        "down": {label: [] for _, _, label in BUCKETS},
    }

    for s in swings:
        label = bucket_for(s["pct_change"])
        if label is None:
            continue  # swing never reached the 1% minimum bucket threshold
        grouped[s["direction"]][label].append(s)

    summary = {}
    for direction, buckets in grouped.items():
        summary[direction] = {}
        for label, items in buckets.items():
            if not items:
                summary[direction][label] = None
                continue
            days = [s["trading_days"] for s in items]
            summary[direction][label] = {
                "count": len(items),
                "median_days": float(np.median(days)),
                "percentiles": {p: float(np.percentile(days, p)) for p in PERCENTILES},
            }
    return summary


def print_report(summary, ticker, reversal_pct):
    for direction in ("up", "down"):
        print(f"\n{'=' * 70}")
        print(f"{ticker} — {direction.upper()} SWINGS  (reversal threshold: {reversal_pct}%)")
        print(f"{'=' * 70}")
        for _, _, label in BUCKETS:
            stats = summary[direction][label]
            print(f"\nBucket: {label}")
            if stats is None:
                print("  No swings in this bucket")
                continue
            print(f"  Number of swings : {stats['count']}")
            print(f"  Median days held : {stats['median_days']:.1f}")
            print("  Percentiles of days held:")
            for p in PERCENTILES:
                print(f"    {p:>3}th : {stats['percentiles'][p]:.1f}")


def write_csv(swings, path):
    fieldnames = [
        "direction", "start_date", "end_date",
        "start_price", "end_price", "pct_change", "trading_days",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in swings:
            row = dict(s)
            row["pct_change"] = round(row["pct_change"], 4)
            writer.writerow(row)


SWING_SUMMARY_FIELDS = ["ticker", "direction", "bucket", "count", "median_days"] + \
    [f"p{p}" for p in PERCENTILES]


def build_swing_summary_rows(ticker, summary):
    """Build one row per direction/bucket of the aggregate stats (count, median days, percentiles)."""
    rows = []
    for direction in ("up", "down"):
        for _, _, label in BUCKETS:
            stats = summary[direction][label]
            row = {"ticker": ticker, "direction": direction, "bucket": label}
            if stats is None:
                row["count"] = 0
                row["median_days"] = ""
                for p in PERCENTILES:
                    row[f"p{p}"] = ""
            else:
                row["count"] = stats["count"]
                row["median_days"] = round(stats["median_days"], 2)
                for p in PERCENTILES:
                    row[f"p{p}"] = round(stats["percentiles"][p], 2)
            rows.append(row)
    return rows


def write_rows_csv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_daily_pct_changes(daily_prices):
    """Return (index, date, pct_change) for every day vs. the prior day's close."""
    changes = []
    for i in range(1, len(daily_prices)):
        prev_price = daily_prices[i - 1][1]
        date, price = daily_prices[i]
        if prev_price <= 0:
            continue  # defensive: load_daily_closes already filters non-positive prices
        pct_change = (price - prev_price) / prev_price * 100
        changes.append((i, date, pct_change))
    return changes


GAP_DIRECTIONS = ["combined", "up", "down"]


def find_swing_day_gaps(daily_changes, threshold_pct, direction="combined"):
    """
    A 'swing day' is any day whose day-over-day close change exceeds
    threshold_pct. direction='up' only counts up days (pct_change > threshold),
    direction='down' only counts down days (pct_change < -threshold), and
    'combined' counts either (|pct_change| > threshold).

    Returns (swing_days, gaps) where gaps is the number of trading days
    between each pair of consecutive same-direction swing days.
    """
    if direction == "up":
        swing_days = [(i, date, pct) for i, date, pct in daily_changes if pct > threshold_pct]
    elif direction == "down":
        swing_days = [(i, date, pct) for i, date, pct in daily_changes if pct < -threshold_pct]
    else:
        swing_days = [(i, date, pct) for i, date, pct in daily_changes if abs(pct) > threshold_pct]

    gaps = [swing_days[k][0] - swing_days[k - 1][0] for k in range(1, len(swing_days))]
    return swing_days, gaps


def summarize_gap_distribution(gaps):
    if not gaps:
        return None
    return {
        "count": len(gaps),
        "median_days": float(np.median(gaps)),
        "percentiles": {p: float(np.percentile(gaps, p)) for p in PERCENTILES},
    }


def analyze_daily_swing_gaps(daily_prices, thresholds):
    """
    For each threshold and each direction ('combined', 'up', 'down'), find
    swing days and the distribution of trading-day gaps between consecutive
    same-direction swing days.

    Returns {threshold: {direction: {'swing_day_count': int, 'gap_stats': dict or None}}}.
    """
    daily_changes = compute_daily_pct_changes(daily_prices)
    results = {}
    for threshold in thresholds:
        results[threshold] = {}
        for direction in GAP_DIRECTIONS:
            swing_days, gaps = find_swing_day_gaps(daily_changes, threshold, direction)
            results[threshold][direction] = {
                "swing_day_count": len(swing_days),
                "gap_stats": summarize_gap_distribution(gaps),
            }
    return results


DIRECTION_LABELS = {
    "combined": "Combined (either direction)",
    "up": "Up swings",
    "down": "Down swings",
}


def print_gap_report(results, ticker):
    print(f"\n{'=' * 70}")
    print(f"{ticker} — DAYS BETWEEN DAILY SWINGS, BY THRESHOLD")
    print(f"{'=' * 70}")
    for threshold in sorted(results):
        print(f"\nDaily swing > {threshold}%")
        for direction in GAP_DIRECTIONS:
            stats = results[threshold][direction]
            print(f"  [{DIRECTION_LABELS[direction]}]")
            print(f"    Number of swing days : {stats['swing_day_count']}")
            gap_stats = stats["gap_stats"]
            if gap_stats is None:
                print("    Not enough swing days to compute gaps (need at least 2)")
                continue
            print(f"    Number of gaps       : {gap_stats['count']}")
            print(f"    Median days between  : {gap_stats['median_days']:.1f}")
            print("    Percentiles of days between swings:")
            for p in PERCENTILES:
                print(f"      {p:>3}th : {gap_stats['percentiles'][p]:.1f}")


GAP_SUMMARY_FIELDS = ["ticker", "threshold_pct", "direction", "swing_day_count", "gap_count",
                      "median_gap_days"] + [f"p{p}" for p in PERCENTILES]


def build_gap_summary_rows(ticker, results):
    """Build one row per threshold/direction of the days-between-swings distribution."""
    rows = []
    for threshold in sorted(results):
        for direction in GAP_DIRECTIONS:
            stats = results[threshold][direction]
            row = {
                "ticker": ticker,
                "threshold_pct": threshold,
                "direction": direction,
                "swing_day_count": stats["swing_day_count"],
            }
            gap_stats = stats["gap_stats"]
            if gap_stats is None:
                row["gap_count"] = 0
                row["median_gap_days"] = ""
                for p in PERCENTILES:
                    row[f"p{p}"] = ""
            else:
                row["gap_count"] = gap_stats["count"]
                row["median_gap_days"] = round(gap_stats["median_days"], 2)
                for p in PERCENTILES:
                    row[f"p{p}"] = round(gap_stats["percentiles"][p], 2)
            rows.append(row)
    return rows


def analyze_ticker(daily_prices, args):
    """Run both analyses (swing buckets + swing-day gaps) for one ticker's daily closes."""
    swings = detect_swings(daily_prices, args.reversal_pct)
    summary = summarize(swings)

    gap_thresholds = [float(t) for t in args.gap_thresholds.split(",") if t.strip()]
    gap_results = analyze_daily_swing_gaps(daily_prices, gap_thresholds)

    return swings, summary, gap_results


def run_single(args, ticker):
    asset_type = args.type or DEFAULT_SINGLE_TYPE
    directory = args.dir or os.path.join(args.dbar_dir, asset_type, subdir_for_ticker(ticker))
    filepath = find_ticker_file(directory, ticker)
    print(f"Loading {filepath} ...")
    daily_prices = load_daily_closes(filepath)
    if len(daily_prices) < 2:
        raise SystemExit("Not enough daily bars found to analyze.")
    print(f"Loaded {len(daily_prices)} trading days "
          f"({daily_prices[0][0]} to {daily_prices[-1][0]})")

    swings, summary, gap_results = analyze_ticker(daily_prices, args)
    print(f"Detected {len(swings)} swings at a {args.reversal_pct}% reversal threshold")

    print_report(summary, ticker, args.reversal_pct)
    print_gap_report(gap_results, ticker)

    if not args.no_summary_out:
        summary_path = args.summary_out or os.path.join(
            DEFAULT_OUTPUT_DIR, f"{ticker}_swing_summary.csv")
        write_rows_csv(build_swing_summary_rows(ticker, summary), SWING_SUMMARY_FIELDS, summary_path)
        print(f"\nWrote aggregated bucket stats to {summary_path}")

    if not args.no_gap_summary_out:
        gap_summary_path = args.gap_summary_out or os.path.join(
            DEFAULT_OUTPUT_DIR, f"{ticker}_swing_gap_summary.csv")
        write_rows_csv(build_gap_summary_rows(ticker, gap_results), GAP_SUMMARY_FIELDS, gap_summary_path)
        print(f"Wrote days-between-swings summary to {gap_summary_path}")

    if args.csv_out:
        write_csv(swings, args.csv_out)
        print(f"Wrote {len(swings)} individual swings to {args.csv_out}")


def run_batch_for_type(args, subdirs, asset_type):
    """Every ticker file for one asset type across the given subdir(s), one CSV pair per subdir."""
    type_processed = 0
    type_skipped = 0

    for subdir in subdirs:
        directory = os.path.join(args.dbar_dir, asset_type, subdir)
        ticker_files = list_ticker_files(directory)
        print(f"\n[{asset_type}/{subdir}] {len(ticker_files)} ticker file(s) found in {directory}")

        swing_rows = []
        gap_rows = []
        processed = 0
        skipped = 0

        for fname in ticker_files:
            ticker = fname.split("_full_1day")[0]
            filepath = os.path.join(directory, fname)
            try:
                daily_prices = load_daily_closes(filepath)
                if len(daily_prices) < MIN_TRADING_DAYS:
                    skipped += 1
                    continue

                _, summary, gap_results = analyze_ticker(daily_prices, args)
                swing_rows.extend(build_swing_summary_rows(ticker, summary))
                gap_rows.extend(build_gap_summary_rows(ticker, gap_results))
                processed += 1
            except Exception as e:
                # Never let one bad ticker (corrupt file, unexpected data) abort
                # the whole multi-thousand-ticker batch run.
                print(f"  [{asset_type}/{subdir}] Skipping {ticker}: {e}")
                skipped += 1
                continue

        summary_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{asset_type}_{subdir}_swing_summary.csv")
        gap_summary_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{asset_type}_{subdir}_swing_gap_summary.csv")
        write_rows_csv(swing_rows, SWING_SUMMARY_FIELDS, summary_path)
        write_rows_csv(gap_rows, GAP_SUMMARY_FIELDS, gap_summary_path)
        print(f"[{asset_type}/{subdir}] Processed {processed}, skipped {skipped} "
              f"(insufficient history or unreadable). Wrote {summary_path} and {gap_summary_path}")

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
    parser = argparse.ArgumentParser(description="Analyze daily up/down swings")
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
    parser.add_argument("--reversal-pct", type=float, default=DEFAULT_REVERSAL_PCT,
                         help="Percent reversal from the running extreme that ends a swing")
    parser.add_argument("--csv-out", help="Single-ticker mode: optional path to write every "
                                           "individual swing as CSV")
    parser.add_argument("--summary-out",
                         help="Single-ticker mode: path for the aggregated bucket stats CSV "
                              f"(default: {DEFAULT_OUTPUT_DIR}/<ticker>_swing_summary.csv)")
    parser.add_argument("--no-summary-out", action="store_true",
                         help="Single-ticker mode: skip writing the aggregate summary CSV")
    parser.add_argument("--gap-thresholds", default=",".join(str(t) for t in DEFAULT_GAP_THRESHOLDS),
                         help="Comma-separated list of daily-swing percent thresholds "
                              f"(default: {','.join(str(t) for t in DEFAULT_GAP_THRESHOLDS)})")
    parser.add_argument("--gap-summary-out",
                         help="Single-ticker mode: path for the days-between-swings CSV "
                              f"(default: {DEFAULT_OUTPUT_DIR}/<ticker>_swing_gap_summary.csv)")
    parser.add_argument("--no-gap-summary-out", action="store_true",
                         help="Single-ticker mode: skip writing the days-between-swings summary CSV")
    args = parser.parse_args()

    if args.ticker:
        run_single(args, args.ticker)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
