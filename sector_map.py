#!/usr/local/bin/python3

import time
import random
import pandas as pd
import yfinance as yf

##############################################################
# Configuration
##############################################################

INPUT_FILE = "/home/ninan/analytics/output/marketcap/filtered.txt"
OUTPUT_FILE = "/home/ninan/analytics/output/marketcap/stock_sector_mapping.csv"

SLEEP_MIN = 0.5
SLEEP_MAX = 1.0
MAX_RETRIES = 3
SAVE_EVERY = 50

##############################################################
# Sector ETF Mapping
##############################################################

SECTOR_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

##############################################################
# Load S&P500 List
##############################################################

print("Loading S&P500 list...")

try:
    sp500 = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]

    SP500 = set(sp500["Symbol"].str.replace(".", "-", regex=False))
except Exception:
    SP500 = set()

##############################################################
# Read tickers
##############################################################

with open(INPUT_FILE) as f:
    tickers = [x.strip() for x in f if x.strip()]

print(f"{len(tickers)} tickers found.")

##############################################################
# Fetch one ticker
##############################################################

def fetch(symbol):

    for attempt in range(MAX_RETRIES):

        try:

            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get("sector", "")
            industry = info.get("industry", "")
            company = info.get("longName", "")
            exchange = info.get("exchange", "")
            market_cap = info.get("marketCap", "")
            sub_industry = info.get("industryDisp", industry)

            return {
                "Ticker": symbol,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "SubIndustry": sub_industry,
                "SectorETF": SECTOR_ETF.get(sector, ""),
                "MarketCap": market_cap,
                "Exchange": exchange,
                "SP500": symbol in SP500
            }

        except Exception as e:

            print(f"{symbol}: attempt {attempt+1} failed ({e})")

            time.sleep(5)

    print(f"{symbol}: FAILED")

    return {
        "Ticker": symbol,
        "Company": "",
        "Sector": "",
        "Industry": "",
        "SubIndustry": "",
        "SectorETF": "",
        "MarketCap": "",
        "Exchange": "",
        "SP500": symbol in SP500
    }

##############################################################
# Main Loop
##############################################################

rows = []

for i, ticker in enumerate(tickers, start=1):

    print(f"{i}/{len(tickers)} : {ticker}")

    rows.append(fetch(ticker))

    if i % SAVE_EVERY == 0:
        pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
        print("Intermediate save complete.")

    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

##############################################################
# Final Save
##############################################################

df = pd.DataFrame(rows)
df.sort_values("Ticker", inplace=True)
df.to_csv(OUTPUT_FILE, index=False)

print()
print("Finished.")
print(df.head())
