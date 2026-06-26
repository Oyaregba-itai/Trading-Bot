"""
Extra sentiment sources:
  - Additional Reddit subreddits via RSS (r/wallstreetbets, r/algotrading, etc.)
  - CoinGecko trending coin sentiment
  - Yahoo Finance RSS news feed
"""
import feedparser
import requests


# Extra subreddits not covered by the main reddit source
_EXTRA_SUBREDDITS = {
    "BTC":    ["BitcoinBeginners", "btc"],
    "ETH":    ["ethereum", "defi"],
    "SOL":    ["solana"],
    "DOGE":   ["dogecoin"],
    "AAPL":   ["wallstreetbets", "investing"],
    "TSLA":   ["wallstreetbets", "teslainvestorsclub"],
    "EURUSD": ["Forex", "algotrading"],
    "GOLD":   ["Gold", "investing"],
    "OIL":    ["investing", "Economics"],
    "_default_crypto": ["CryptoCurrency", "CryptoMarkets", "altcoin"],
    "_default_stock":  ["wallstreetbets", "stocks", "investing"],
}


def fetch_extra_reddit(symbol: str, limit: int = 20) -> list[str]:
    """Extra Reddit subreddits via RSS — returns list of title+summary texts."""
    sym = symbol.upper()
    from config import CRYPTO_IDS
    if sym in _EXTRA_SUBREDDITS:
        subs = _EXTRA_SUBREDDITS[sym]
    elif sym in CRYPTO_IDS:
        subs = _EXTRA_SUBREDDITS["_default_crypto"]
    else:
        subs = _EXTRA_SUBREDDITS["_default_stock"]

    texts = []
    for sub in subs[:2]:
        try:
            feed = feedparser.parse(f"https://www.reddit.com/r/{sub}/hot.rss?limit={limit}")
            for e in feed.entries[:limit]:
                title = e.get("title", "")
                if sym.lower() in title.lower() or not texts:   # include general if no matches
                    texts.append(title)
        except Exception:
            continue
    return texts[:30]


def fetch_yahoo_finance_news(symbol: str, limit: int = 10) -> list[str]:
    """
    Yahoo Finance RSS news feed.
    Works for stocks, ETFs, crypto (BTC-USD), and forex (EURUSD=X).
    """
    sym = symbol.upper()
    _ticker_map = {
        "BTC":    "BTC-USD",
        "ETH":    "ETH-USD",
        "SOL":    "SOL-USD",
        "BNB":    "BNB-USD",
        "DOGE":   "DOGE-USD",
        "EURUSD": "EURUSD=X",
        "GOLD":   "GC=F",
        "OIL":    "CL=F",
    }
    ticker = _ticker_map.get(sym, sym)

    try:
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(url)
        return [e.get("title", "") for e in feed.entries[:limit] if e.get("title")]
    except Exception:
        return []


def fetch_seeking_alpha_rss(symbol: str, limit: int = 8) -> list[str]:
    """Seeking Alpha RSS for stocks and crypto."""
    try:
        url = f"https://seekingalpha.com/api/sa/combined/{symbol.lower()}.xml"
        feed = feedparser.parse(url)
        return [e.get("title", "") for e in feed.entries[:limit] if e.get("title")]
    except Exception:
        return []
