#!/usr/bin/env python3
"""
Download the FRED 10-year Treasury constant maturity yield (series DGS10) at
daily, weekly, monthly and annual frequency.

The daily file is the raw series. The weekly (weeks ending Friday), monthly and
annual files use FRED's own frequency aggregation, so the values match what
the FRED website shows for those frequencies. Rows with no value (market
holidays, and the current period when FRED leaves it blank) are dropped.

If an output file already exists only the gap is downloaded: the series is
fetched from the file's last date onward (FRED still aggregates that whole
period, so a partial week/month/year is refreshed) and merged in. Use --full
to re-download everything.

Source (no API key needed):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10[&fq=...&fam=...]

Output:
    /home/ninan/analytics/data/rates/<SERIES>_daily.txt
    /home/ninan/analytics/data/rates/<SERIES>_weekly.txt
    /home/ninan/analytics/data/rates/<SERIES>_monthly.txt
    /home/ninan/analytics/data/rates/<SERIES>_annual.txt
    date,<SERIES>

Usage:
    ./fred_rates.py                  # DGS10, period average
    ./fred_rates.py --agg eop        # DGS10, end-of-period value
    ./fred_rates.py --series DGS2    # any other FRED daily series
    ./fred_rates.py --full           # ignore existing files, download all
"""
import argparse
import io
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

OUT_DIR = '/home/ninan/analytics/data/rates'
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

# name -> FRED fq parameter (None = native daily frequency)
FREQUENCIES = [
    ('daily', None),
    ('weekly', 'Weekly, Ending Friday'),
    ('monthly', 'Monthly'),
    ('annual', 'Annual'),
]


def fetch(series, fq, agg, start=None):
    params = {'id': series}
    if start:
        params['cosd'] = start
    if fq:
        params['fq'] = fq
        params['fam'] = agg
    url = FRED_URL + '?' + urllib.parse.urlencode(params)
    # FRED stalls on some custom/browser User-Agents; urllib's default works
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(text))
    df.columns = ['date', series]
    df[series] = pd.to_numeric(df[series], errors='coerce')
    return df.dropna(subset=[series])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--series', default='DGS10', help='FRED series id (default DGS10)')
    ap.add_argument('--agg', choices=['avg', 'eop'], default='avg',
                    help='aggregation for weekly/monthly/annual: period average '
                         'or end of period (default avg)')
    ap.add_argument('--outdir', default=OUT_DIR)
    ap.add_argument('--full', action='store_true',
                    help='re-download the whole series instead of only the gap')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for name, fq in FREQUENCIES:
        path = os.path.join(args.outdir, f'{args.series}_{name}.txt')
        old = None
        if not args.full and os.path.exists(path):
            old = pd.read_csv(path, dtype={'date': str})
            if old.empty:
                old = None
        start = old['date'].iloc[-1] if old is not None else None
        try:
            new = fetch(args.series, fq, args.agg, start)
        except Exception as e:
            print(f'{args.series} {name}: download failed: {e}', file=sys.stderr)
            return 1
        if old is not None:
            # new rows replace any old rows from the re-fetched periods
            if not new.empty:
                old = old[old['date'] < new['date'].iloc[0]]
            df = pd.concat([old, new], ignore_index=True)
            added = len(df) - len(old)
        else:
            df = new
            added = len(df)
        df.to_csv(path, index=False)
        print(f'{path}: {len(df)} rows, {df["date"].iloc[0]} .. {df["date"].iloc[-1]}'
              f' ({added} fetched{" since " + start if start else ""})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
