Quant: Market Data Analysis Toolkit

A set of Python tools for studying how U.S. stocks and ETFs behave, built on years of 1-minute and daily price data.

Why I built it:
I wanted to answer specific questions about market behavior with data instead of intuition. For example: *When SPY drops 1% in the first half hour, what does a given stock usually do? How long does a typical 5% swing last?* The toolkit has two parts: an interactive charting app for exploring the data visually, and a set of analysis scripts that run across the whole stock and ETF universe.

Interactive Candlestick Chart (candlechart.py)
A Plotly Dash web app for exploring 1-minute OHLCV data.
- Zoom-aware resampling: keeps the full history in memory and picks the candle size automatically (1-minute up to monthly) based on zoom level, so years of data stay fast to browse.
- Multi-ticker overlays: charts a stock alongside SPY, a sector ETF and the 10-year Treasury yield on one synced time axis. Panning or zooming any chart moves them all.
- Smooth panning: loads data on both sides of the visible range, so ordinary pans are handled in the browser without a server call.
- Remote data sync: fetches data files from a remote server over SSH/SCP, re-copies them when the server's copy is newer, and falls back to the local copy when the server is offline.

Analysis & Data Scripts:
The other scripts make up a batch pipeline that collects market data and measures how stocks behave:
- Volatility: rolling and annualized realized volatility, and Average True Range (ATR).
- Price swings: splits price history into up-swings and down-swings, and measures how big they get, how long they last, and how much time passes between large daily moves.
- Moving with the market: for each half hour of the trading day, measures how often a stock moves the same way as SPY, QQQ, DIA and Treasuries (IEF), and how much it moves.
- Data collection: downloads Treasury yield series from the Federal Reserve's FRED, and maps each ticker to its sector and matching SPDR sector ETF.

All the scripts can process every ticker in the universe, with the data split into letter-pair folders so the work can run in parallel. Downloads only fetch new data, and API calls retry on failure. Results are saved as CSV summaries and optional charts.
