#!/usr/bin/env python3
"""
Interactive candlestick chart for a ticker's 1-minute OHLCV data.

The whole history is held in memory and the chart is served from it, so panning
and zooming keep finding data instead of running off the end of a fixed window:

  * Drag to pan, scroll to zoom, double-click to fit the whole history.
  * Draw a box with the modebar's "Box Select" tool, then press
    "Zoom to selection" to zoom into exactly that time and price range.
  * "Auto" candle size picks the finest candle that keeps the view readable, so
    zooming out to years of data stays fast (it lands on daily, weekly or
    monthly candles) and
    zooming in reveals 1-minute detail.
  * Each render loads a margin either side of what is on screen, so ordinary
    pans are served by the browser with no round trip to the server.
  * The date pickers set the starting window; after that they are just a way to
    jump somewhere specific.

Runs on either box:
  * On a trader box the data is already local, so nothing is copied.
  * On the Mac the ticker file is copied with scp from a trader box into
    ~/analytics/data/stocks/<shard>/ the first time it is charted, then the
    browser opens on the chart. On later runs the server's copy is checked over
    ssh and copied again if it is newer, so a file copied days ago does not
    hide the latest bars. If the server cannot be reached the local copy is used.

Data files live in letter-pair shards, e.g. AA -> <data-dir>/ab/AA_full_1min_adjsplit.txt.
ETFs use the same layout under ~/analytics/data/etfs, e.g. SPY -> etfs/st/. A
ticker is looked for among the stocks first, then the ETFs, locally and then on
the trader box, so ./candlechart.py SPY needs no extra option.

Treasury series names (DGS10, DGS2, DGS3MO, ...) and any name with a file in
~/analytics/data/rates are charted as a daily rate series instead, e.g. DGS10 ->
~/analytics/data/rates/DGS10_daily.txt (date,value per line), copied and kept
fresh from the trader box the same way; --rates forces this for other names.
The daily values play the part the 1-minute bars play for stocks: a weekly or
monthly candle opens at the period's first value, closes at its last, and its
high and low are the extreme values within it. With only one value per day, a
daily candle opens at the previous day's value so its body shows the day's move.

Several tickers can be charted together to study how macro moves (rates, the
index, a sector) line up with a stock: the first ticker is the primary, the rest
are secondaries. The top chart overlays all their candles, the primary in
green/red on the price scale and each secondary in its own colour, scaled to
fill the chart (hover for its prices); each ticker then gets its own candle
chart below. All charts share one time window,
candle size and session, so panning or zooming any of them moves them all.

Usage:
    ./candlechart.py AA
    ./candlechart.py AA SPY XLE DGS10   # AA overlaid with the index, a sector and rates
    ./candlechart.py SPY                # an ETF, found under data/etfs
    ./candlechart.py AA --data-dir ~/analytics/data/stocks --port 8050
    ./candlechart.py AA --remote ninan@192.168.1.96:/home/ninan/analytics/data/stocks
    ./candlechart.py AA --no-fetch      # use only the local copy; never contact the trader box
    ./candlechart.py AA --no-browser    # just print the URL
    ./candlechart.py DGS10              # a rate series; Daily/Weekly/Monthly in the dropdown

The remote defaults can also be set once via the CANDLECHART_REMOTE,
CANDLECHART_ETF_REMOTE and CANDLECHART_RATES_REMOTE env vars.

Requires: pip install dash plotly pandas
"""

import os
import re
import shlex
import string
import argparse
import subprocess
import threading
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, ALL, ctx, no_update

DEFAULT_DATA_DIR = os.path.expanduser("~/analytics/data/stocks")
DEFAULT_REMOTE = os.environ.get(
    "CANDLECHART_REMOTE", "ninan@192.168.1.98:/home/ninan/analytics/data/stocks")
FILE_SUFFIX = "_full_1min_adjsplit.txt"
DEFAULT_ETF_DIR = os.path.expanduser("~/analytics/data/etfs")
DEFAULT_ETF_REMOTE = os.environ.get(
    "CANDLECHART_ETF_REMOTE", "ninan@192.168.1.98:/home/ninan/analytics/data/etfs")
DEFAULT_RATES_DIR = os.path.expanduser("~/analytics/data/rates")
DEFAULT_RATES_REMOTE = os.environ.get(
    "CANDLECHART_RATES_REMOTE", "ninan@192.168.1.98:/home/ninan/analytics/data/rates")
RATES_SUFFIX = "_daily.txt"
# FRED Treasury constant-maturity series: DGS1MO, DGS3MO, DGS1, DGS10, DGS30, ...
RATE_SERIES_PATTERN = re.compile(r"^DGS\d+(MO)?$")

# Candle count that fills the plot at a readable density; "Auto" aims for this
TARGET_CANDLES = 400
# Steps on the candle ladder are wide (10 min -> 30 min), so let "Auto" run a
# little over target rather than give up that much detail
TARGET_SLACK = 1.2
# Plotting more candles than this makes the browser crawl
MAX_CANDLES = 20000

# Extra span loaded either side of the visible window, as a multiple of its
# width. Pans within the margin are handled by the browser alone.
BUFFER_FACTOR = 0.6
# The visible window must stay inside this share of the loaded span, otherwise
# the server reloads around the new position.
INNER_FRACTION = 0.8
# Zoom is re-served once the visible span moves outside this band, which also
# refits the price axis to what is on screen.
SPAN_TOLERANCE = (0.7, 1.4)

AUTO = "Auto"
INTERVAL_RULES = {
    "1 min": "1min",
    "5 min": "5min",
    "10 min": "10min",
    "30 min": "30min",
    "1 hour": "1h",
    "Daily": "1D",
    "Weekly": "1W",
    "Monthly": "1M",
}
RULE_LABELS = {rule: label for label, rule in INTERVAL_RULES.items()}
# Coarsest last; "Auto" walks this in order. Daily counts as one session, a
# week as five and a month as a typical 21.
RULE_LADDER = ["1min", "5min", "10min", "30min", "1h", "1D", "1W", "1M"]
RULE_MINUTES = {"1min": 1, "5min": 5, "10min": 10, "30min": 30, "1h": 60,
                "1D": 390, "1W": 390 * 5, "1M": 390 * 21}
# Candles of a day or longer: no time of day, no overnight breaks
DAY_OR_LONGER = {"1D", "1W", "1M"}
# pandas period codes for the candles built from daily ones
PERIOD_CODES = {"1W": "W", "1M": "M"}
# Opening window for each starting candle size
DEFAULT_LOOKBACK = {"Daily": pd.DateOffset(years=1), "Weekly": pd.DateOffset(years=5),
                    "Monthly": None}


def shard_for(ticker):
    """
    Return the shard directory for a ticker: letters are paired a-b, c-d, ..., y-z.
    """
    first = ticker[0].lower()
    idx = string.ascii_lowercase.index(first)
    start = idx - (idx % 2)
    return string.ascii_lowercase[start:start + 2]


def remote_mtime(remote_path):
    """
    Modification time (epoch seconds) of a host:path file on the trader box, or
    None if the box cannot be reached or the file is not there.
    """
    host, path = remote_path.split(':', 1)
    try:
        # BatchMode so a missing key fails instead of prompting; a short timeout
        # so an unreachable box only costs a few seconds
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
             f"stat -c %Y {shlex.quote(path)}"],
            capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def copy_from_remote(remote_path, local_path):
    """
    scp remote_path to local_path, returning True on success.

    The copy lands in a .part file that is renamed into place only once it is
    complete, so a failed copy never clobbers a good (if stale) local file.
    """
    part_path = local_path + ".part"
    # -q off so the transfer progress is visible; these files can be hundreds of MB
    result = subprocess.run(["scp", "-C", remote_path, part_path])
    if result.returncode != 0 or not os.path.exists(part_path):
        if os.path.exists(part_path):
            os.remove(part_path)
        return False
    os.replace(part_path, local_path)
    print(f"Copied {os.path.getsize(local_path) / 1e6:,.1f} MB to {local_path}")
    return True


def format_mtime(epoch):
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M')


def stock_paths(ticker, data_dir, remote):
    """
    Local and remote paths of a ticker's 1-minute file, inside its shard.
    """
    shard = shard_for(ticker)
    file_name = f"{ticker}{FILE_SUFFIX}"
    return (os.path.join(data_dir, shard, file_name),
            f"{remote.rstrip('/')}/{shard}/{file_name}")


def find_ticker_paths(ticker, fetch):
    """
    Local and remote paths of a ticker's 1-minute file, looking among the
    stocks first and then the ETFs.

    A local copy decides it without a network call; otherwise the trader box is
    asked which of the two it has. Falls back to the stock paths so the usual
    "not found" error names them.
    """
    candidates = [stock_paths(ticker, DEFAULT_DATA_DIR, DEFAULT_REMOTE),
                  stock_paths(ticker, DEFAULT_ETF_DIR, DEFAULT_ETF_REMOTE)]
    for local_path, remote_path in candidates:
        if os.path.exists(local_path):
            return local_path, remote_path
    if fetch:
        for local_path, remote_path in candidates:
            if remote_mtime(remote_path) is not None:
                return local_path, remote_path
    return candidates[0]


def rates_paths(series, data_dir, remote):
    """
    Local and remote paths of a daily rate series file, e.g. DGS10_daily.txt.
    """
    file_name = f"{series}{RATES_SUFFIX}"
    return os.path.join(data_dir, file_name), f"{remote.rstrip('/')}/{file_name}"


def is_rate_series(name, rates_dir):
    """
    True for names charted as a daily rate series rather than 1-minute stock data.
    """
    return bool(RATE_SERIES_PATTERN.match(name)) or \
        os.path.exists(os.path.join(rates_dir, f"{name}{RATES_SUFFIX}"))


def ensure_local_file(local_path, remote_path, fetch=True):
    """
    Return local_path, fetching it with scp from remote_path if it is missing
    or older than the copy on the trader box.
    """
    if os.path.exists(local_path):
        if not fetch:
            return local_path
        # On the trader box itself the local file is the server's file
        if os.path.realpath(local_path) == os.path.normpath(remote_path.split(':', 1)[1]):
            return local_path

        local_time = os.path.getmtime(local_path)
        server_time = remote_mtime(remote_path)
        if server_time is None:
            print(f"Could not check {remote_path} for newer data; "
                  f"using local copy from {format_mtime(local_time)}")
            return local_path
        if server_time <= local_time:
            return local_path

        print(f"Local copy ({format_mtime(local_time)}) is older than the server's "
              f"({format_mtime(server_time)}), refreshing from {remote_path}")
        if not copy_from_remote(remote_path, local_path):
            print(f"Refresh failed; using local copy from {format_mtime(local_time)}")
        return local_path

    if not fetch:
        raise SystemExit(f"{local_path} not found and --no-fetch was given")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"{local_path} not found, copying from {remote_path}")
    if not copy_from_remote(remote_path, local_path):
        raise SystemExit(
            f"scp of {remote_path} failed.\n"
            f"Check that the trader box is reachable (ssh {remote_path.split(':')[0]}) "
            f"and that the file exists there.\n"
            f"Use --remote to point at a different box, e.g.\n"
            f"  --remote ninan@192.168.1.96:/home/ninan/analytics/data/stocks")
    return local_path


def read_stock_file(file_path):
    """
    Read a headerless timestamp,open,high,low,close,volume file into a DataFrame
    indexed by timestamp.
    """
    df = pd.read_csv(file_path, header=None,
                     names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def read_rates_file(file_path):
    """
    Read a date,value daily series into one-point bars (open = high = low =
    close = the day's value), the daily counterpart of 1-minute stock bars.
    Volume is left empty.
    """
    df = pd.read_csv(file_path)
    df.columns = ['timestamp', 'close']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # FRED marks holidays with "." in raw downloads
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna().set_index('timestamp').sort_index()
    df['open'] = df['high'] = df['low'] = df['close']
    df['volume'] = float('nan')
    return df[['open', 'high', 'low', 'close', 'volume']]


def resample(df, rule, regular_hours):
    """
    Aggregate 1-minute bars into bars of the given pandas frequency rule.
    """
    if rule == "1min":
        return df

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    if rule in DAY_OR_LONGER:
        out = df.groupby(df.index.normalize()).agg(agg)
        if rule in PERIOD_CODES:
            # Label each week or month with its first trading day, which is
            # always a real date on the axis (a calendar start may be a weekend)
            periods = out.index.to_period(PERIOD_CODES[rule])
            first_days = out.index.to_series().groupby(periods).first()
            out = out.groupby(periods).agg(agg)
            out.index = pd.DatetimeIndex(first_days.values)
    elif rule == "1h" and regular_hours:
        # Align hourly bars to the 9:30 open instead of the top of the hour
        out = df.resample(rule, offset="30min", label='left', closed='left').agg(agg)
    else:
        out = df.resample(rule, label='left', closed='left').agg(agg)

    out = out.dropna(subset=['open'])
    if df['volume'].isna().all():
        # A series with no volume (rates) stays empty rather than summing to 0
        out['volume'] = float('nan')
    return out


class BarSource:
    """
    Serves resampled bars for arbitrary windows of a 1-minute DataFrame.

    Slicing goes through searchsorted so picking a window out of a multi-million
    row frame stays O(log n), and recently built windows are cached so panning
    back over ground already covered costs nothing.
    """

    CACHE_LIMIT = 8

    def __init__(self, df, daily=False):
        self.df = df
        self.start = df.index.min()
        self.end = df.index.max()
        # A daily series stands in for a whole session per row and has no
        # intraday candles to offer
        self.daily = daily
        self.row_minutes = RULE_MINUTES["1D"] if daily else 1
        self.ladder = [r for r in RULE_LADDER if r in DAY_OR_LONGER] if daily else RULE_LADDER
        # Share of rows inside the regular session, used to size candles without
        # having to filter the frame first
        n = len(df)
        if daily or not n:
            self.regular_ratio = 1.0
        else:
            self.regular_ratio = len(df.between_time('09:30', '15:59')) / n
        # Weekly and monthly candles aggregate the one-point bars like any other
        # source, but a one-point daily candle would be a flat dash, so daily
        # candles open at the previous day's value to show the day's move
        if daily:
            day = df.copy()
            day['open'] = df['close'].shift(1).fillna(df['close'])
            day['high'] = day[['open', 'close']].max(axis=1)
            day['low'] = day[['open', 'close']].min(axis=1)
            self.day_candles = day
        self._cache = {}

    def minute_count(self, start, end, regular_hours):
        """
        Estimated number of 1-minute bars between start and end.
        """
        lo = self.df.index.searchsorted(start, side='left')
        hi = self.df.index.searchsorted(end, side='right')
        n = max(hi - lo, 0) * self.row_minutes
        return n * self.regular_ratio if regular_hours else n

    def rule_for(self, start, end, regular_hours):
        """
        Finest candle size that keeps the window near TARGET_CANDLES candles.
        """
        minutes = self.minute_count(start, end, regular_hours)
        for rule in self.ladder:
            if minutes / RULE_MINUTES[rule] <= TARGET_CANDLES * TARGET_SLACK:
                return rule
        return self.ladder[-1]

    def bars(self, start, end, rule, regular_hours):
        """
        Resampled bars for [start, end], cached by window.
        """
        key = (start.isoformat(), end.isoformat(), rule, regular_hours)
        if key in self._cache:
            return self._cache[key]

        lo = self.df.index.searchsorted(start, side='left')
        hi = self.df.index.searchsorted(end, side='right')
        frame = self.day_candles if self.daily and rule == "1D" else self.df
        window = frame.iloc[lo:hi]
        if regular_hours:
            window = window.between_time('09:30', '15:59')
        bars = resample(window, rule, regular_hours)

        if len(self._cache) >= self.CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = bars
        return bars


def clamp_window(source, start, end):
    """
    Keep a requested window inside the data, preserving its width where possible.
    """
    span = end - start
    if span <= pd.Timedelta(0):
        span = pd.Timedelta(days=1)
    full = source.end - source.start

    if span >= full:
        return source.start, source.end
    if start < source.start:
        start, end = source.start, source.start + span
    if end > source.end:
        start, end = source.end - span, source.end
    return max(start, source.start), min(end, source.end)


def load_window(source, vis_start, vis_end):
    """
    Widen the visible window by the buffer margin, clipped to the data bounds.
    """
    pad = (vis_end - vis_start) * BUFFER_FACTOR
    return max(vis_start - pad, source.start), min(vis_end + pad, source.end)


def visible_range(relayout):
    """
    Pull the new x range out of a Graph relayoutData payload.

    Returns a (start, end) pair, the string 'all' for a reset to autorange, or
    None for interactions that leave the x range alone.
    """
    if not relayout:
        return None

    if relayout.get('xaxis.autorange') or relayout.get('xaxis2.autorange'):
        return 'all'

    for axis in ('xaxis', 'xaxis2'):
        lo, hi = relayout.get(f'{axis}.range[0]'), relayout.get(f'{axis}.range[1]')
        if lo is None or hi is None:
            pair = relayout.get(f'{axis}.range')
            if pair:
                lo, hi = pair[0], pair[1]
        if lo is not None and hi is not None:
            try:
                return pd.Timestamp(lo), pd.Timestamp(hi)
            except (ValueError, TypeError):
                return None

    return None


def selection_range(selected):
    """
    Pull the box drawn with the Box Select tool out of a Graph selectedData payload.

    Returns (start, end, price_range) where price_range is None when the box was
    drawn on the volume panel, or None if there is no box.
    """
    box = (selected or {}).get('range') or {}
    xs = box.get('x') or box.get('x2')
    if not xs or len(xs) != 2:
        return None
    try:
        start, end = sorted(pd.Timestamp(x) for x in xs)
    except (ValueError, TypeError):
        return None
    if end <= start:
        return None
    ys = box.get('y')
    price_range = (min(ys), max(ys)) if ys and len(ys) == 2 else None
    return start, end, price_range


def served_by_browser(source, view, vis_start, vis_end, label, rule, regular_hours):
    """
    True when the loaded figure already covers this view, so the pan or zoom
    needs no new figure from the server.
    """
    if not view or view['label'] != label or view['regular'] != regular_hours:
        return False
    if view['rule'] != rule:
        return False

    prev_start, prev_end = pd.Timestamp(view['vis_start']), pd.Timestamp(view['vis_end'])
    prev_span = prev_end - prev_start
    if prev_span <= pd.Timedelta(0):
        return False

    # Plotly echoes the range we set when the figure is replaced; treat that as
    # unchanged so it does not bounce straight back for another render
    tol = prev_span * 0.01
    if abs(vis_start - prev_start) <= tol and abs(vis_end - prev_end) <= tol:
        return True

    ratio = (vis_end - vis_start) / prev_span
    if not SPAN_TOLERANCE[0] <= ratio <= SPAN_TOLERANCE[1]:
        return False

    load_start, load_end = pd.Timestamp(view['load_start']), pd.Timestamp(view['load_end'])
    margin = (load_end - load_start) * (1 - INNER_FRACTION) / 2
    # A side that already sits on the edge of the data cannot be extended, so it
    # counts as covered however close the view gets to it
    left_ok = vis_start >= load_start + margin or load_start <= source.start
    right_ok = vis_end <= load_end - margin or load_end >= source.end
    return left_ok and right_ok


def snap_to_bars(bars, vis_start, vis_end):
    """
    Slide a window that landed in a gap onto the nearest bars.

    Zooming or panning can leave the view sitting entirely inside an overnight
    break, a weekend or a holiday, which would draw a blank chart. Keep the
    width and move it to where the data is.
    """
    if bars.empty:
        return vis_start, vis_end

    lo = bars.index.searchsorted(vis_start, side='left')
    hi = bars.index.searchsorted(vis_end, side='right')
    if hi > lo:
        return vis_start, vis_end

    span = vis_end - vis_start
    if lo >= len(bars):
        return bars.index[-1] - span, bars.index[-1]
    return bars.index[lo], bars.index[lo] + span


def rangebreaks_for(bars, rule, regular_hours):
    """
    Hide weekends, missing trading days (holidays) and overnight hours on the x axis.
    """
    if rule in PERIOD_CODES:
        # Gaps of a few days do not matter at this scale, and listing every
        # non-candle weekday as a break would be thousands of entries
        return []

    if rule in DAY_OR_LONGER:
        breaks = [dict(bounds=["sat", "mon"])]
        if len(bars) > 0:
            trading_days = pd.DatetimeIndex(bars.index.normalize().unique())
            all_weekdays = pd.bdate_range(trading_days.min(), trading_days.max())
            holidays = all_weekdays.difference(trading_days)
            if len(holidays) > 0:
                breaks.append(dict(values=holidays.strftime('%Y-%m-%d').tolist()))
        return breaks

    # Intraday: plotly misplaces bars when breaks overlap, as a weekend, a
    # Monday holiday and the overnight hours all do around Labor Day (Sep 1-4
    # drew on top of Sep 8-11). So keep them disjoint: the overnight break
    # covers close to next open, and a day without trading loses only its
    # session hours, which together tile every gap exactly once.
    open_hour, close_hour = (9.5, 16) if regular_hours else (4, 20)
    breaks = [dict(bounds=[close_hour, open_hour], pattern="hour")]
    if len(bars) == 0:
        return breaks

    trading_days = pd.DatetimeIndex(bars.index.normalize().unique())
    first, last = trading_days.min(), trading_days.max()
    # Run a couple of weeks past the bars so weekends stay hidden when panning
    # to the edge of what is loaded
    days = pd.date_range(first - pd.Timedelta(days=14), last + pd.Timedelta(days=14), freq='D')
    inside = (days >= first) & (days <= last)
    closed = days[(days.dayofweek >= 5) | (inside & ~days.isin(trading_days))]
    if len(closed) > 0:
        starts = closed + pd.Timedelta(hours=open_hour)
        breaks.append(dict(values=starts.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                           dvalue=(close_hour - open_hour) * 3600 * 1000))
    return breaks


def axis_ranges(bars, vis_start, vis_end):
    """
    Price and volume ranges fitted to the bars inside the visible window, so
    panning to a different price level does not push the candles off screen.
    """
    lo = bars.index.searchsorted(vis_start, side='left')
    hi = bars.index.searchsorted(vis_end, side='right')
    visible = bars.iloc[lo:hi]
    if visible.empty:
        visible = bars
    if visible.empty:
        return None, None

    low, high = visible['low'].min(), visible['high'].max()
    if pd.isna(low) or pd.isna(high):
        return None, None
    low, high = float(low), float(high)
    pad = (high - low) * 0.05 or max(abs(high) * 0.01, 0.01)

    volume = visible['volume'].max()
    volume = 0.0 if pd.isna(volume) else float(volume)
    return (low - pad, high + pad), (0, volume * 1.15 or 1)


def format_window(start, end, rule):
    fmt = '%Y-%m-%d' if rule in DAY_OR_LONGER else '%Y-%m-%d %H:%M'
    return f"{start.strftime(fmt)} to {end.strftime(fmt)}"


def build_figure(bars, ticker, rule, regular_hours, vis_start, vis_end, label, note,
                 price_range=None, value_label="Price", height=800):
    # A series with no volume (rates) gets the whole height for its candles
    show_volume = bars['volume'].notna().any()
    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.78, 0.22])
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars['open'], high=bars['high'],
        low=bars['low'], close=bars['close'], name=ticker,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ), row=1, col=1)

    if show_volume:
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for o, c in zip(bars['open'], bars['close'])]
        fig.add_trace(go.Bar(x=bars.index, y=bars['volume'], marker_color=colors,
                             name='Volume', showlegend=False), row=2, col=1)

    size = RULE_LABELS.get(rule, rule)
    if label == AUTO:
        size += " (auto)"
    title = f"{ticker} &nbsp;·&nbsp; {size} &nbsp;·&nbsp; {format_window(vis_start, vis_end, rule)}"
    subtitle = note or ("Drag to pan · scroll to zoom · double-click to fit all · "
                        "Box Select then 'Zoom to selection' to zoom into a box")
    title += f"<br><sup>{subtitle}</sup>"

    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=80, b=40),
        hovermode='x unified',
        # Plotly drags a zoom box by default; panning is what this chart is for
        dragmode='pan',
        # Keep the browser's own pan/zoom state out of the way of the range we set
        uirevision=f"{vis_start.isoformat()}|{vis_end.isoformat()}|{rule}|{regular_hours}",
    )
    fig.update_xaxes(rangebreaks=rangebreaks_for(bars, rule, regular_hours),
                     range=[vis_start, vis_end])

    fitted_price, volume_range = axis_ranges(bars, vis_start, vis_end)
    price_range = price_range or fitted_price
    fig.update_yaxes(title_text=value_label, range=price_range, row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", range=volume_range, row=2, col=1)
    return fig


def create_app(ticker, df, daily=False, interval=AUTO):
    source = BarSource(df, daily=daily)
    first_day = source.start.date()
    last_day = source.end.date()
    if interval in DEFAULT_LOOKBACK:
        lookback = DEFAULT_LOOKBACK[interval]
    else:
        lookback = pd.DateOffset(years=1) if daily else pd.DateOffset(months=1)
    default_start = first_day if lookback is None else max(first_day, (source.end - lookback).date())
    sizes = [AUTO] + [k for k, rule in INTERVAL_RULES.items() if rule in source.ladder]
    value_label = "Rate (%)" if daily else "Price"
    kind = "daily" if daily else "1-minute"

    app = Dash(__name__, title=f"{ticker} candles")
    app.layout = html.Div([
        html.H3(f"{ticker}: {kind} data from {first_day} to {last_day}"),
        html.Div([
            html.Label("Jump to "),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=first_day,
                max_date_allowed=last_day,
                start_date=default_start,
                end_date=last_day,
                display_format='YYYY-MM-DD',
            ),
            html.Label("  Candle size ", style={'marginLeft': '20px'}),
            dcc.Dropdown(
                id='interval',
                options=[{'label': k, 'value': k} for k in sizes],
                value=interval if interval in sizes else AUTO,
                clearable=False,
                style={'width': '140px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            ),
            dcc.Checklist(
                id='session',
                options=[{'label': ' Regular hours only (9:30-16:00)', 'value': 'regular'}],
                value=[] if daily else ['regular'],
                # A daily series has no session hours to filter
                style={'display': 'none' if daily else 'inline-block', 'marginLeft': '20px'},
            ),
            html.Button('Zoom to selection', id='zoom-selection', disabled=True,
                        title='Draw a box with the Box Select tool first',
                        style={'marginLeft': '20px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '6px'}),
        # delay_show keeps the spinner from flashing on the quick renders
        dcc.Loading(dcc.Graph(id='chart', config={'scrollZoom': True}), delay_show=400),
        dcc.Store(id='view'),
    ], style={'fontFamily': 'sans-serif', 'margin': '16px'})

    @app.callback(
        Output('zoom-selection', 'disabled'),
        Input('chart', 'selectedData'),
    )
    def toggle_zoom_button(selected):
        return selection_range(selected) is None

    @app.callback(
        Output('chart', 'figure'),
        Output('view', 'data'),
        Output('chart', 'selectedData'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('interval', 'value'),
        Input('session', 'value'),
        Input('chart', 'relayoutData'),
        Input('zoom-selection', 'n_clicks'),
        State('chart', 'selectedData'),
        State('view', 'data'),
    )
    def update_chart(start_date, end_date, label, session, relayout, _clicks, selected, view):
        regular_hours = 'regular' in (session or []) and not daily
        price_range = None
        # The drawn box disappears with the redraw, so drop its stale selection too
        clear_selection = no_update

        if ctx.triggered_id == 'chart':
            wanted = visible_range(relayout)
            if wanted is None:
                # A y-axis drag, a hover change, the initial autosize: nothing to redraw
                return no_update, no_update, no_update
            vis_start, vis_end = ((source.start, source.end) if wanted == 'all' else wanted)
        elif ctx.triggered_id == 'zoom-selection':
            box = selection_range(selected)
            if box is None:
                return no_update, no_update, no_update
            vis_start, vis_end, price_range = box
            clear_selection = None
        elif ctx.triggered_id in ('interval', 'session') and view:
            # Changing candle size or session should stay where the user is looking
            vis_start = pd.Timestamp(view['vis_start'])
            vis_end = pd.Timestamp(view['vis_end'])
        else:
            vis_start = pd.Timestamp(start_date or first_day)
            vis_end = pd.Timestamp(end_date or last_day) + pd.Timedelta(days=1)

        vis_start, vis_end = clamp_window(source, vis_start, vis_end)

        note = ""
        minutes = source.minute_count(vis_start, vis_end, regular_hours)
        if label == AUTO:
            rule = source.rule_for(vis_start, vis_end, regular_hours)
        else:
            rule = INTERVAL_RULES[label]
            # A fine candle over a wide window would hand the browser millions of
            # candles, so fall back to the auto size rather than wedge the page
            if minutes / RULE_MINUTES[rule] > MAX_CANDLES:
                rule = source.rule_for(vis_start, vis_end, regular_hours)
                note = (f"{label} candles would be too many to draw over this range; "
                        f"showing {RULE_LABELS[rule]} instead.")

        if ctx.triggered_id == 'chart' and served_by_browser(
                source, view, vis_start, vis_end, label, rule, regular_hours):
            return no_update, no_update, no_update

        def around(start, end):
            # A view already dense enough to be slow to draw is not worth a margin
            if source.minute_count(start, end, regular_hours) / RULE_MINUTES[rule] > 2 * TARGET_CANDLES:
                return start, end
            return load_window(source, start, end)

        load_start, load_end = around(vis_start, vis_end)
        bars = source.bars(load_start, load_end, rule, regular_hours)

        if bars.empty:
            # The whole window sits in a gap; cast wider to find the nearest bars
            load_start = max(vis_start - pd.Timedelta(days=10), source.start)
            load_end = min(vis_end + pd.Timedelta(days=10), source.end)
            bars = source.bars(load_start, load_end, rule, regular_hours)

        snapped = snap_to_bars(bars, vis_start, vis_end)
        if snapped != (vis_start, vis_end):
            vis_start, vis_end = snapped
            load_start, load_end = around(vis_start, vis_end)
            bars = source.bars(load_start, load_end, rule, regular_hours)

        if bars.empty and not note:
            note = "No data in this range."

        fig = build_figure(bars, ticker, rule, regular_hours,
                           vis_start, vis_end, label, note, price_range, value_label)
        new_view = {
            'vis_start': vis_start.isoformat(), 'vis_end': vis_end.isoformat(),
            'load_start': load_start.isoformat(), 'load_end': load_end.isoformat(),
            'rule': rule, 'label': label, 'regular': regular_hours,
        }
        return fig, new_view, clear_selection

    return app

# Candle colours for the secondaries on the overlay, clear of the primary's green/red
OVERLAY_COLORS = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
# Height of each chart when several tickers are on the page
MULTI_HEIGHT = 650


class Series:
    """
    One ticker on a multi-ticker page: its name, its bars and how to label it.
    """

    def __init__(self, name, df, daily):
        self.name = name
        self.source = BarSource(df, daily=daily)
        self.daily = daily
        self.value_label = f"{name} rate (%)" if daily else f"{name} price"

    def rule(self, rule):
        """
        The candle size this series can offer for the page's candle size; a daily
        series has nothing finer than a day.
        """
        return rule if rule in self.source.ladder else self.source.ladder[0]


def faded(hex_color, alpha=0.2):
    """
    rgba() form of a #rrggbb colour at the given opacity.
    """
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def session_candles(series, bars, color, axis, regular_hours):
    """
    Daily candles drawn across their whole session on an intraday axis.

    A daily series (rates) on the overlay of an intraday view would otherwise
    be one thin candle per day lost among the intraday ones, so each day's
    candle is drawn as a bar body spanning the day's session, with a thin bar
    for the wick, matching the daily candles in the series' own chart.
    """
    open_hour, close_hour = (9.5, 16) if regular_hours else (4, 20)
    session_ms = (close_hour - open_hour) * 3600 * 1000
    mid = bars.index + pd.Timedelta(hours=(open_hour + close_hour) / 2)

    rising = bars['close'] >= bars['open']
    body_low = bars[['open', 'close']].min(axis=1)
    body = (bars['close'] - bars['open']).abs()
    # An unchanged day would have no body at all; give it a sliver to see
    sliver = (bars['high'].max() - bars['low'].min()) * 0.004 or 0.001
    body = body.clip(lower=sliver)

    hover = (f"{series.name} O %{{customdata[0]:.2f}} H %{{customdata[1]:.2f}} "
             f"L %{{customdata[2]:.2f}} C %{{customdata[3]:.2f}}<extra></extra>")
    # The wick is drawn above and below the body only, so it does not show
    # through a hollow body
    body_high = bars[['open', 'close']].max(axis=1)
    wicks = [go.Bar(x=mid, y=top - bottom, base=bottom, width=session_ms * 0.03,
                    marker=dict(color=color, line=dict(width=0)), yaxis=axis,
                    legendgroup=series.name, showlegend=False, hoverinfo='skip')
             for bottom, top in ((body_high, bars['high']), (bars['low'], body_low))]
    candle = go.Bar(x=mid, y=body, base=body_low, width=session_ms * 0.8,
                    marker=dict(color=[faded(color) if r else color for r in rising],
                                line=dict(color=color, width=1)),
                    yaxis=axis, name=series.name, legendgroup=series.name,
                    customdata=bars[['open', 'high', 'low', 'close']].values,
                    hovertemplate=hover)
    return [*wicks, candle], bars.set_index(mid)


def build_overlay_figure(primary, primary_bars, secondaries, rule, regular_hours,
                         vis_start, vis_end, label, note, price_range=None):
    """
    Every ticker's candles on one chart: the primary in green/red on the left
    price scale, each secondary in its own colour on its own scale. The
    secondaries' scales are hidden to keep the full width for the candles;
    hovering shows each ticker's prices.

    secondaries is a list of (Series, bars) pairs.
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=primary_bars.index, open=primary_bars['open'], high=primary_bars['high'],
        low=primary_bars['low'], close=primary_bars['close'], name=primary.name,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ))

    axes = {}
    for k, (series, bars) in enumerate(secondaries):
        color = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
        axis = f"y{k + 2}"
        if series.daily and rule not in DAY_OR_LONGER and not bars.empty:
            traces, bars = session_candles(series, bars, color, axis, regular_hours)
            fig.add_traces(traces)
        else:
            # Up candles hollow and down candles solid, so direction reads
            # without the green/red that belongs to the primary
            fig.add_trace(go.Candlestick(
                x=bars.index, open=bars['open'], high=bars['high'],
                low=bars['low'], close=bars['close'], name=series.name, yaxis=axis,
                increasing=dict(line=dict(color=color), fillcolor=faded(color)),
                decreasing=dict(line=dict(color=color), fillcolor=color),
            ))
        fitted, _ = axis_ranges(bars, vis_start, vis_end)
        axes[f"yaxis{k + 2}"] = dict(overlaying='y', range=fitted, visible=False)

    size = RULE_LABELS.get(rule, rule)
    if label == AUTO:
        size += " (auto)"
    names = ", ".join(series.name for series, _ in secondaries)
    title = (f"{primary.name} with {names} &nbsp;·&nbsp; {size} &nbsp;·&nbsp; "
             f"{format_window(vis_start, vis_end, rule)}")
    subtitle = note or (f"Price scale is {primary.name}'s; each other ticker is scaled to fill "
                        f"the chart (up hollow, down solid) · hover for prices · "
                        f"drag to pan · scroll to zoom")
    title += f"<br><sup>{subtitle}</sup>"

    fitted_price, _ = axis_ranges(primary_bars, vis_start, vis_end)
    fig.update_layout(
        title=title,
        height=MULTI_HEIGHT,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
        xaxis=dict(rangeslider_visible=False,
                   rangebreaks=rangebreaks_for(primary_bars, rule, regular_hours),
                   range=[vis_start, vis_end]),
        yaxis=dict(title=dict(text=primary.value_label), range=price_range or fitted_price),
        margin=dict(l=60, r=30, t=90, b=40),
        hovermode='x unified',
        # Session-wide daily candles are bars; keep them on top of one another
        barmode='overlay',
        dragmode='pan',
        uirevision=f"{vis_start.isoformat()}|{vis_end.isoformat()}|{rule}|{regular_hours}",
        **axes,
    )
    return fig


def create_multi_app(series_list, interval=AUTO):
    """
    One page for several tickers: an overlay of them all on top, then each
    ticker's own candle chart, all sharing one time window.

    The first series is the primary; it sets the window bounds, the candle
    size and the gaps hidden on the overlay's time axis.
    """
    primary = series_list[0]
    secondaries = series_list[1:]
    source = primary.source
    first_day = source.start.date()
    last_day = source.end.date()
    if interval in DEFAULT_LOOKBACK:
        lookback = DEFAULT_LOOKBACK[interval]
    else:
        lookback = pd.DateOffset(years=1) if primary.daily else pd.DateOffset(months=1)
    default_start = first_day if lookback is None else max(first_day, (source.end - lookback).date())
    sizes = [AUTO] + [k for k, rule in INTERVAL_RULES.items() if rule in source.ladder]
    names = [series.name for series in series_list]
    kind = "daily" if primary.daily else "1-minute"

    # Chart 0 is the overlay; chart i + 1 is series i on its own
    def chart(index):
        return dcc.Loading(dcc.Graph(id={'type': 'chart', 'index': index},
                                     config={'scrollZoom': True}), delay_show=400)

    app = Dash(__name__, title=f"{' / '.join(names)} candles")
    app.layout = html.Div([
        html.H3(f"{primary.name} ({kind} data from {first_day} to {last_day}) "
                f"with {', '.join(names[1:])}"),
        html.Div([
            html.Label("Jump to "),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=first_day,
                max_date_allowed=last_day,
                start_date=default_start,
                end_date=last_day,
                display_format='YYYY-MM-DD',
            ),
            html.Label("  Candle size ", style={'marginLeft': '20px'}),
            dcc.Dropdown(
                id='interval',
                options=[{'label': k, 'value': k} for k in sizes],
                value=interval if interval in sizes else AUTO,
                clearable=False,
                style={'width': '140px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            ),
            dcc.Checklist(
                id='session',
                options=[{'label': ' Regular hours only (9:30-16:00)', 'value': 'regular'}],
                value=[] if primary.daily else ['regular'],
                style={'display': 'none' if primary.daily else 'inline-block', 'marginLeft': '20px'},
            ),
            html.Button('Zoom to selection', id='zoom-selection', disabled=True,
                        title='Draw a box with the Box Select tool first',
                        style={'marginLeft': '20px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '6px'}),
        chart(0),
        html.H4("Individual charts", style={'marginTop': '24px'}),
        *[chart(i + 1) for i in range(len(series_list))],
        dcc.Store(id='view'),
        dcc.Store(id='selection'),
    ], style={'fontFamily': 'sans-serif', 'margin': '16px'})

    n_charts = len(series_list) + 1

    @app.callback(
        Output('selection', 'data'),
        Input({'type': 'chart', 'index': ALL}, 'selectedData'),
    )
    def remember_selection(_selected):
        # Whichever chart a box was last drawn on is the one to zoom into
        if not isinstance(ctx.triggered_id, dict):
            return no_update
        selected = ctx.triggered[0]['value']
        if selection_range(selected) is None:
            return None
        return {'index': ctx.triggered_id['index'], 'selected': selected}

    @app.callback(
        Output('zoom-selection', 'disabled'),
        Input('selection', 'data'),
    )
    def toggle_zoom_button(selection):
        return not selection

    @app.callback(
        Output({'type': 'chart', 'index': ALL}, 'figure'),
        Output('view', 'data'),
        Output({'type': 'chart', 'index': ALL}, 'selectedData'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('interval', 'value'),
        Input('session', 'value'),
        Input({'type': 'chart', 'index': ALL}, 'relayoutData'),
        Input('zoom-selection', 'n_clicks'),
        State('selection', 'data'),
        State('view', 'data'),
    )
    def update_charts(start_date, end_date, label, session, _relayouts, _clicks, selection, view):
        unchanged = [no_update] * n_charts, no_update, [no_update] * n_charts
        regular_hours = 'regular' in (session or []) and not primary.daily
        price_range, box_chart = None, None
        clear_selection = [no_update] * n_charts
        trigger = ctx.triggered_id

        if isinstance(trigger, dict):
            wanted = visible_range(ctx.triggered[0]['value'])
            if wanted is None:
                return unchanged
            vis_start, vis_end = (source.start, source.end) if wanted == 'all' else wanted
            if view and wanted != 'all':
                # Redrawn charts echo back the range they were given; that is
                # not a move, and redrawing again would loop
                prev_start, prev_end = pd.Timestamp(view['vis_start']), pd.Timestamp(view['vis_end'])
                tol = (prev_end - prev_start) * 0.01
                if abs(vis_start - prev_start) <= tol and abs(vis_end - prev_end) <= tol:
                    return unchanged
        elif trigger == 'zoom-selection':
            box = selection_range((selection or {}).get('selected'))
            if box is None:
                return unchanged
            vis_start, vis_end, price_range = box
            box_chart = selection['index']
            clear_selection = [None] * n_charts
        elif trigger in ('interval', 'session') and view:
            vis_start = pd.Timestamp(view['vis_start'])
            vis_end = pd.Timestamp(view['vis_end'])
        else:
            vis_start = pd.Timestamp(start_date or first_day)
            vis_end = pd.Timestamp(end_date or last_day) + pd.Timedelta(days=1)

        vis_start, vis_end = clamp_window(source, vis_start, vis_end)

        note = ""
        minutes = source.minute_count(vis_start, vis_end, regular_hours)
        if label == AUTO:
            rule = source.rule_for(vis_start, vis_end, regular_hours)
        else:
            rule = INTERVAL_RULES[label]
            if minutes / RULE_MINUTES[rule] > MAX_CANDLES:
                rule = source.rule_for(vis_start, vis_end, regular_hours)
                note = (f"{label} candles would be too many to draw over this range; "
                        f"showing {RULE_LABELS[rule]} instead.")

        def around(start, end):
            if source.minute_count(start, end, regular_hours) / RULE_MINUTES[rule] > 2 * TARGET_CANDLES:
                return start, end
            return load_window(source, start, end)

        load_start, load_end = around(vis_start, vis_end)
        primary_bars = source.bars(load_start, load_end, rule, regular_hours)
        if primary_bars.empty:
            load_start = max(vis_start - pd.Timedelta(days=10), source.start)
            load_end = min(vis_end + pd.Timedelta(days=10), source.end)
            primary_bars = source.bars(load_start, load_end, rule, regular_hours)

        snapped = snap_to_bars(primary_bars, vis_start, vis_end)
        if snapped != (vis_start, vis_end):
            vis_start, vis_end = snapped
            load_start, load_end = around(vis_start, vis_end)
            primary_bars = source.bars(load_start, load_end, rule, regular_hours)

        def series_bars(series):
            if series is primary:
                return primary_bars
            own_rule = series.rule(rule)
            start = load_start
            if own_rule != rule:
                # A daily series in an intraday view: reach back far enough to
                # have the close that holds at the start of the window
                start = load_start - pd.Timedelta(days=7)
            return series.source.bars(start, load_end, own_rule,
                                      regular_hours and not series.daily)

        all_bars = [series_bars(series) for series in series_list]

        figures = [build_overlay_figure(
            primary, primary_bars, list(zip(secondaries, all_bars[1:])), rule, regular_hours,
            vis_start, vis_end, label, note, price_range if box_chart == 0 else None)]
        for i, (series, bars) in enumerate(zip(series_list, all_bars)):
            own_rule = series.rule(rule)
            own_note = note
            if bars.empty:
                own_note = f"No {series.name} data in this range."
            elif own_rule != rule:
                own_note = f"{series.name} has one value per day, so it is shown as daily candles."
            figures.append(build_figure(
                bars, series.name, own_rule, regular_hours and not series.daily,
                vis_start, vis_end, label if own_rule == rule else "", own_note,
                price_range if box_chart == i + 1 else None,
                series.value_label, height=MULTI_HEIGHT))

        new_view = {'vis_start': vis_start.isoformat(), 'vis_end': vis_end.isoformat(),
                    'rule': rule, 'label': label, 'regular': regular_hours}
        return figures, new_view, clear_selection

    return app


def load_series(ticker, args):
    """
    Find, fetch if needed, and read one ticker's data. Returns (df, is_rate_series).
    """
    rates = args.rates or is_rate_series(
        ticker, os.path.expanduser(args.data_dir or DEFAULT_RATES_DIR))
    if rates:
        data_dir = os.path.expanduser(args.data_dir or DEFAULT_RATES_DIR)
        local_path, remote_path = rates_paths(ticker, data_dir, args.remote or DEFAULT_RATES_REMOTE)
    elif args.data_dir or args.remote:
        # An explicit location means just that location, stock or ETF
        data_dir = os.path.expanduser(args.data_dir or DEFAULT_DATA_DIR)
        local_path, remote_path = stock_paths(ticker, data_dir, args.remote or DEFAULT_REMOTE)
    else:
        local_path, remote_path = find_ticker_paths(ticker, fetch=not args.no_fetch)
    file_path = ensure_local_file(local_path, remote_path, fetch=not args.no_fetch)

    print(f"Loading {file_path}")
    df = read_rates_file(file_path) if rates else read_stock_file(file_path)
    if df.empty:
        raise SystemExit(f"{file_path} has no rows")
    kind = "daily" if rates else "1-minute"
    print(f"Loaded {len(df):,} {kind} bars from {df.index.min()} to {df.index.max()}")
    return df, rates


def main():
    parser = argparse.ArgumentParser(description='Interactive candlestick chart from 1-minute OHLCV data.')
    parser.add_argument('tickers', nargs='+', metavar='ticker',
                        help='Ticker symbol, e.g. AA, or a rate series, e.g. DGS10. Give several '
                             'to overlay them: the first is the primary, the rest are secondaries')
    parser.add_argument('--rates', action='store_true',
                        help=f'Chart every ticker as a daily rate series <ticker>{RATES_SUFFIX} even if the '
                             f'name is not recognised as one (DGS10 etc. are detected automatically)')
    parser.add_argument('--data-dir',
                        help=f'Local data directory (default {DEFAULT_DATA_DIR}, then {DEFAULT_ETF_DIR}, '
                             f'or {DEFAULT_RATES_DIR} for a rate series)')
    parser.add_argument('--remote',
                        help=f'scp source for missing or stale files (default {DEFAULT_REMOTE}, '
                             f'then {DEFAULT_ETF_REMOTE}, '
                             f'or {DEFAULT_RATES_REMOTE} for a rate series)')
    parser.add_argument('--interval', default=AUTO, choices=[AUTO, *INTERVAL_RULES],
                        help='Candle size to open with (default Auto)')
    parser.add_argument('--no-fetch', action='store_true',
                        help='Never contact the trader box: use the local file as is, '
                             'and fail if the ticker is not already local')
    parser.add_argument('--port', type=int, default=8050, help='Port for the web UI (default 8050)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open a browser; just print the URL')
    args = parser.parse_args()

    # dict.fromkeys drops repeats while keeping the order, so the primary stays first
    tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    if len(tickers) == 1:
        df, rates = load_series(tickers[0], args)
        app = create_app(tickers[0], df, daily=rates, interval=args.interval)
    else:
        series_list = []
        for ticker in tickers:
            df, rates = load_series(ticker, args)
            series_list.append(Series(ticker, df, rates))
        app = create_multi_app(series_list, interval=args.interval)

    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving {', '.join(tickers)} at {url}  (ctrl-c to stop)")
    if not args.no_browser:
        # The server blocks below, so hand the browser off to a timer
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    try:
        app.run(debug=False, port=args.port)
    except OSError as exc:
        raise SystemExit(f"Could not serve on port {args.port}: {exc}\n"
                         f"Another chart may already be running; try --port {args.port + 1}")


if __name__ == "__main__":
    main()
