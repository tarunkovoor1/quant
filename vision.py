#!/usr/local/bin/python3

"""
Vision - Per-ticker intraday index/treasury co-movement matrix (1-minute bars).

Co-movement is inherently a per-ticker relationship, so this is a single-pass
script (no reduce step): for every ticker it builds, over that ticker's whole
history, a matrix describing how the ticker moves when each of four reference
instruments moves by a given amount over the same interval.

Reference instruments (loaded into memory once and evaluated side by side):
    SPY  (S&P 500 ETF)
    QQQ  (Nasdaq-100 ETF)
    DIA  (Dow Jones Industrial Average ETF)
    IEF  (7-10yr Treasury ETF)

Intervals
---------
The 13 half-hour intervals of the trading day, plus a whole-day aggregate:

    0930-1000, 1000-1030, 1030-1100, 1100-1130, 1130-1200, 1200-1230,
    1230-1300, 1300-1330, 1330-1400, 1400-1430, 1430-1500, 1500-1530,
    1530-1600, and DAY  (previous close -> today's close)

For the first half-hour (0930-1000) and for DAY the "start" price is the
PREVIOUS trading day's close (so the overnight gap is included); every other
half hour measures start-time -> end-time.  Each move is
(end - start) / start * 100.

The matrix (per ticker)
-----------------------
For every (reference instrument, interval, reference direction up/down,
reference-move magnitude bucket) the script records, using the ticker's move
over that same interval:

  * direction agreement - how often the ticker moved the SAME direction as the
    reference vs. the OPPOSITE direction (and how often it was flat), and
  * the distribution of the ticker's signed % move - mean and percentiles
    p1, p5, p10, p15, p25, median(p50), p75, p85, p90, p95, p99.

Reference move magnitude buckets
--------------------------------
SPY / QQQ / DIA (percent):
    0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0, 1.0-1.5, 1.5-2.0, 2.0-3.0,
    3.0-4.0, 4.0-5.0, 5.0-10.0, 10.0-20.0, 20.0+
IEF (percent):
    0-0.1, 0.1-0.2, ... 0.9-1.0, 1.0-1.25, 1.25-1.5, 1.5-1.75, 1.75-2.0,
    2.0-2.5, 2.5-3.0, 3.0-4.0, 4.0-5.0, 5.0+

Usage:
    python3 vision.py --type TYPE --subdir SUBDIR [SUBDIR ...]

    Designed to be launched across ticker ranges via run_all.sh, e.g.:
        ./run_all.sh vision.py first
        ./run_all.sh vision.py second

Output:
    One file per subdirectory written to /home/ninan/analytics/output/vision
        {TYPE}_{SUBDIR}_vision.txt

    Columns:
        ticker, index, interval, index_direction, index_bucket,
        bucket_low, bucket_high, n, n_same, n_opposite, n_flat,
        pct_same, pct_opposite, pct_flat, mean_stock_pct,
        p1, p5, p10, p15, p25, median, p75, p85, p90, p95, p99

    Only buckets that actually occurred for a ticker are emitted (empty buckets
    are omitted).  Percentile columns are blank when a bucket has fewer than
    MIN_OBS observations.
"""

import os
import csv
import bisect
import argparse

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BASE_DATA_DIR = '/home/ninan/analytics/data'
ETF_DIR = os.path.join(BASE_DATA_DIR, 'etfs')      # reference instruments live here
OUTPUT_DIR = '/home/ninan/analytics/output/vision'

INDEX_TICKERS = ['SPY', 'QQQ', 'DIA', 'IEF']

# Interval boundary marks, expressed as minutes-since-midnight.
#   09:30 = 570, 10:00 = 600, ... 15:59 = 959 (the 16:00 close is the 15:59 bar)
MARK_MINUTES = [600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 930, 959]

# 13 half-hour interval labels + a whole-day aggregate appended at the end.
INTERVAL_LABELS = [
    '0930-1000', '1000-1030', '1030-1100', '1100-1130', '1130-1200',
    '1200-1230', '1230-1300', '1300-1330', '1330-1400', '1400-1430',
    '1430-1500', '1500-1530', '1530-1600', 'DAY',
]
DAY_IDX = len(INTERVAL_LABELS) - 1                 # index of the DAY aggregate

MARKET_OPEN_MINUTE = 570        # 09:30 - earliest usable bar within a day

# Percentiles reported for the ticker's signed % move within each bucket.
PERCENTILES = [1, 5, 10, 15, 25, 50, 75, 85, 90, 95, 99]
PCT_LABELS = ['p1', 'p5', 'p10', 'p15', 'p25', 'median',
              'p75', 'p85', 'p90', 'p95', 'p99']

MIN_OBS = 20               # minimum observations before percentiles are reported

OUTPUT_FIELDS = (['ticker', 'index', 'interval', 'index_direction',
                  'index_bucket', 'bucket_low', 'bucket_high', 'n',
                  'n_same', 'n_opposite', 'n_flat',
                  'pct_same', 'pct_opposite', 'pct_flat', 'mean_stock_pct']
                 + PCT_LABELS)


# --------------------------------------------------------------------------
# Magnitude buckets
# --------------------------------------------------------------------------
def build_std_buckets():
    """Magnitude buckets for SPY/QQQ/DIA - list of (low, high, label)."""
    edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
    buckets = [(lo, hi, f'{lo}-{hi}') for lo, hi in zip(edges[:-1], edges[1:])]
    buckets.append((20.0, float('inf'), '20+'))
    return buckets


def build_ief_buckets():
    """Magnitude buckets for IEF - list of (low, high, label)."""
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
             1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    buckets = [(lo, hi, f'{lo}-{hi}') for lo, hi in zip(edges[:-1], edges[1:])]
    buckets.append((5.0, float('inf'), '5+'))
    return buckets


STD_BUCKETS = build_std_buckets()
IEF_BUCKETS = build_ief_buckets()
STD_LOWS = [b[0] for b in STD_BUCKETS]
IEF_LOWS = [b[0] for b in IEF_BUCKETS]


def buckets_for(index):
    return IEF_BUCKETS if index == 'IEF' else STD_BUCKETS


def bucket_index(index, mag):
    """Index of the magnitude bucket for a non-negative reference move."""
    lows = IEF_LOWS if index == 'IEF' else STD_LOWS
    i = bisect.bisect_right(lows, mag) - 1
    return i if i >= 0 else 0


# --------------------------------------------------------------------------
# Data location / parsing
# --------------------------------------------------------------------------
def first_letter_subdir(ticker):
    """Return the two-letter data subdirectory for a ticker (a,b -> 'ab', ...)."""
    pairs = ['ab', 'cd', 'ef', 'gh', 'ij', 'kl', 'mn',
             'op', 'qr', 'st', 'uv', 'wx', 'yz']
    c = ticker[0].lower()
    for p in pairs:
        if c in p:
            return p
    return None


def find_ticker_file(base_dir, subdir, ticker):
    """Locate the 1-minute data file for a specific ticker inside a subdir."""
    path = os.path.join(base_dir, subdir)
    if not os.path.isdir(path):
        return None
    prefix = f"{ticker}_"
    for f in os.listdir(path):
        if f.startswith(prefix) and '_1min_' in f:
            return os.path.join(path, f)
    return None


def price_at_or_before(minutes, closes, target):
    """Close of the latest bar at/before target minute (>= market open), else None."""
    idx = bisect.bisect_right(minutes, target) - 1
    if idx >= 0 and minutes[idx] >= MARKET_OPEN_MINUTE:
        return closes[idx]
    return None


def extract_minute_prices(ticker_file):
    """
    Parse a 1-minute data file into per-day price structures.

    Data rows: 'YYYY-MM-DD HH:MM:SS,open,high,low,close,volume' (chronological).

    Returns:
        (dates_order, day_bars, day_close)
            dates_order: list of date strings in chronological order (unique)
            day_bars:    date -> (minutes_list, closes_list)   (time-sorted)
            day_close:   date -> close of the last bar of the day
    """
    dates_order = []
    day_bars = {}
    day_close = {}
    last_date = None

    try:
        with open(ticker_file, 'r') as f:
            for line in f:
                if len(line) < 19:
                    continue
                ts = line[:19]                      # 'YYYY-MM-DD HH:MM:SS'
                try:
                    date_str = ts[:10]
                    minute = int(ts[11:13]) * 60 + int(ts[14:16])
                    close = float(line[20:].split(',', 4)[3])
                except (ValueError, IndexError):
                    continue

                if date_str != last_date:
                    if date_str not in day_bars:
                        day_bars[date_str] = ([], [])
                        dates_order.append(date_str)
                    last_date = date_str

                mins, closes = day_bars[date_str]
                mins.append(minute)
                closes.append(close)
                day_close[date_str] = close
    except Exception as e:
        print(f"  Error reading {ticker_file}: {e}")
        return [], {}, {}

    return dates_order, day_bars, day_close


def compute_interval_pcts(day_bars, day_close, dates_order):
    """
    Compute the 13 half-hour interval moves plus the DAY move for every date.

    Returns:
        date -> list of 14 floats (or None where an interval is unusable):
                indices 0..12 = half-hour intervals, index 13 = DAY.
    """
    prev_close_of = {}
    prev = None
    for d in dates_order:
        prev_close_of[d] = day_close.get(prev) if prev is not None else None
        prev = d

    out = {}
    for d in dates_order:
        mins, closes = day_bars[d]
        marks = [price_at_or_before(mins, closes, m) for m in MARK_MINUTES]
        prev_close = prev_close_of[d]

        starts = [prev_close] + marks[:-1]
        ends = marks
        pcts = []
        for s, e in zip(starts, ends):
            if s is not None and e is not None and s > 0:
                pcts.append((e - s) / s * 100.0)
            else:
                pcts.append(None)

        # DAY aggregate: previous close -> today's close.
        today_close = day_close.get(d)
        if prev_close is not None and today_close is not None and prev_close > 0:
            pcts.append((today_close - prev_close) / prev_close * 100.0)
        else:
            pcts.append(None)

        out[d] = pcts
    return out


# --------------------------------------------------------------------------
# Matrix builder
# --------------------------------------------------------------------------
def _pct(count, total):
    return round(count / total * 100.0, 4) if total else 0.0


def build_ticker_rows(ticker, stock_pcts, index_pcts):
    """
    Build the matrix rows for one ticker.

    Args:
        ticker:      ticker symbol
        stock_pcts:  date -> [14 interval pcts] for the ticker
        index_pcts:  {index -> {date -> [14 interval pcts]}} for the references

    Returns:
        list of output-row dicts.
    """
    # cell key: (index, interval_idx, direction, bucket_idx) -> list of moves
    cells = {}
    for d, spct in stock_pcts.items():
        for index in INDEX_TICKERS:
            ip_map = index_pcts.get(index)
            if not ip_map:
                continue
            ipct = ip_map.get(d)
            if ipct is None:
                continue
            for k in range(len(INTERVAL_LABELS)):
                sp = spct[k]
                ip = ipct[k]
                if sp is None or ip is None or ip == 0.0:
                    continue
                direction = 'up' if ip > 0 else 'down'
                b_idx = bucket_index(index, abs(ip))
                cells.setdefault((index, k, direction, b_idx), []).append(sp)

    rows = []
    for (index, k, direction, b_idx), moves in cells.items():
        arr = np.asarray(moves, dtype=np.float64)
        n = arr.size
        if direction == 'up':
            n_same = int(np.count_nonzero(arr > 0))
            n_opp = int(np.count_nonzero(arr < 0))
        else:
            n_same = int(np.count_nonzero(arr < 0))
            n_opp = int(np.count_nonzero(arr > 0))
        n_flat = int(np.count_nonzero(arr == 0))

        lo, hi, label = buckets_for(index)[b_idx]
        row = {
            'ticker': ticker,
            'index': index,
            'interval': INTERVAL_LABELS[k],
            'index_direction': direction,
            'index_bucket': label,
            'bucket_low': lo,
            'bucket_high': 'inf' if hi == float('inf') else hi,
            'n': n,
            'n_same': n_same,
            'n_opposite': n_opp,
            'n_flat': n_flat,
            'pct_same': _pct(n_same, n),
            'pct_opposite': _pct(n_opp, n),
            'pct_flat': _pct(n_flat, n),
            'mean_stock_pct': round(float(arr.mean()), 4),
        }
        if n >= MIN_OBS:
            pvals = np.percentile(arr, PERCENTILES)
            for lbl, v in zip(PCT_LABELS, pvals):
                row[lbl] = round(float(v), 4)
        else:
            for lbl in PCT_LABELS:
                row[lbl] = ''
        rows.append(row)

    # Stable ordering: index, interval, direction, bucket.
    dir_order = {'up': 0, 'down': 1}
    rows.sort(key=lambda r: (
        INDEX_TICKERS.index(r['index']),
        INTERVAL_LABELS.index(r['interval']),
        dir_order[r['index_direction']],
        [b[2] for b in buckets_for(r['index'])].index(r['index_bucket']),
    ))
    return rows


class VisionMapper:
    """Builds the per-ticker co-movement matrix for one subdirectory."""

    def __init__(self, sec_type, subdir):
        self.sec_type = sec_type
        self.subdir = subdir
        self.base_dir = os.path.join(BASE_DATA_DIR, sec_type)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.index_pcts = {}       # index -> {date -> [14 pcts]}

    def load_indexes(self):
        """Load and pre-compute per-day interval moves for the reference set."""
        for idx in INDEX_TICKERS:
            sd = first_letter_subdir(idx)
            path = find_ticker_file(ETF_DIR, sd, idx)
            if path is None:
                print(f"  WARNING: reference instrument {idx} not found "
                      f"(looked in {os.path.join(ETF_DIR, sd)}) - its rows "
                      f"will be absent")
                self.index_pcts[idx] = {}
                continue
            dates_order, day_bars, day_close = extract_minute_prices(path)
            self.index_pcts[idx] = compute_interval_pcts(
                day_bars, day_close, dates_order)
            print(f"  loaded reference {idx}: {len(dates_order)} days")

    def get_ticker_list(self):
        """Full paths to all 1-minute ticker files in the subdir."""
        ticker_path = os.path.join(self.base_dir, self.subdir)
        if not os.path.exists(ticker_path):
            print(f"  Directory {ticker_path} does not exist")
            return []
        return [
            os.path.join(ticker_path, f)
            for f in os.listdir(ticker_path)
            if os.path.isfile(os.path.join(ticker_path, f)) and '_1min_' in f
        ]

    def process_ticker(self, ticker_file):
        """Return the matrix rows for one ticker."""
        ticker = os.path.basename(ticker_file).split('_full_1min')[0]
        dates_order, day_bars, day_close = extract_minute_prices(ticker_file)
        if not dates_order:
            return []
        stock_pcts = compute_interval_pcts(day_bars, day_close, dates_order)
        return build_ticker_rows(ticker, stock_pcts, self.index_pcts)

    def run(self):
        output_file = os.path.join(
            OUTPUT_DIR, f"{self.sec_type}_{self.subdir}_vision.txt")

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()

        ticker_files = self.get_ticker_list()
        if not ticker_files:
            print(f"  No ticker files found for {self.sec_type}/{self.subdir}")
            return output_file

        total_rows = 0
        processed = 0
        for ticker_file in ticker_files:
            try:
                rows = self.process_ticker(ticker_file)
            except Exception as e:
                print(f"  Error processing {ticker_file}: {e}")
                continue
            if rows:
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_FIELDS,
                                            extrasaction='ignore')
                    writer.writerows(rows)
                total_rows += len(rows)
            processed += 1

        print(f"  {self.sec_type}/{self.subdir}: processed {processed} tickers, "
              f"wrote {total_rows} rows to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Vision - per-ticker intraday index/treasury co-movement matrix')
    parser.add_argument('--type', choices=['stocks', 'etfs'], required=True,
                        help='Type of securities (stocks or etfs)')
    parser.add_argument('--subdir', required=True, nargs='+',
                        help='One or more subdirectories to process (ab, cd, ...)')
    args = parser.parse_args()

    print(f"Running Vision for {args.type} in subdirectories: "
          f"{', '.join(args.subdir)}")
    print(f"Output will be written to: {OUTPUT_DIR}")

    print("\nLoading reference instruments (SPY, QQQ, DIA, IEF) into memory...")
    # The reference set is the same for every subdir, so load once and reuse.
    shared = VisionMapper(args.type, args.subdir[0])
    shared.load_indexes()

    for subdir in args.subdir:
        print(f"\nProcessing subdirectory: {subdir}")
        mapper = VisionMapper(args.type, subdir)
        mapper.index_pcts = shared.index_pcts
        mapper.run()

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
