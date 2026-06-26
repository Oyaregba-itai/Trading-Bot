"""
Reddit via RSS feeds — no authentication required.
Reddit blocked unauthenticated JSON API in 2023; RSS still works.
"""
import feedparser
import requests

_SUBREDDIT_MAP = {
    "BTC":    ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"],
    "ETH":    ["ethereum", "CryptoCurrency", "ethtrader"],
    "SOL":    ["solana", "CryptoCurrency"],
    "BNB":    ["binance", "CryptoCurrency"],
    "DOGE":   ["dogecoin", "CryptoCurrency"],
    "PEPE":   ["pepecoin", "memecoins", "CryptoCurrency"],
    "XRP":    ["Ripple", "CryptoCurrency"],
    "ADA":    ["cardano", "CryptoCurrency"],
    "AAPL":   ["stocks", "investing", "StockMarket"],
    "TSLA":   ["teslainvestorsclub", "stocks", "investing"],
    "EURUSD": ["Forex", "investing"],
    "GOLD":   ["Gold", "investing"],
    "OIL":    ["investing", "Economics"],
}

_DEFAULT_CRYPTO   = ["CryptoCurrency", "CryptoMarkets"]
_DEFAULT_STOCK    = ["stocks", "investing", "StockMarket"]
_DEFAULT_FOREX    = ["Forex", "investing"]


def _get_subreddits(symbol: str) -> list[str]:
    from config import CRYPTO_IDS, COMMODITY_SYMBOLS
    sym = symbol.upper()
    if sym in _SUBREDDIT_MAP:
        return _SUBREDDIT_MAP[sym]
    if sym in CRYPTO_IDS:
        return _DEFAULT_CRYPTO
    if sym in COMMODITY_SYMBOLS or len(sym) == 6:
        return _DEFAULT_FOREX
    return _DEFAULT_STOCK


def fetch_reddit_posts(subreddit: str, query: str = "", sort: str = "hot", limit: int = 20) -> list[dict]:
    """Fetch posts via Reddit RSS (no auth)."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.rss?limit={limit}"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:limit]:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            # If query specified, filter by keyword
            if query and query.lower() not in (title + summary).lower():
                continue
            results.append({
                "title": title,
                "text":  summary[:300],
                "score": 0,
                "url":   entry.get("link", ""),
            })
        return results
    except Exception:
        return []


def fetch_symbol_reddit(symbol: str, limit: int = 20) -> list[dict]:
    """Collect posts from all relevant subreddits for a symbol."""
    subreddits = _get_subreddits(symbol)
    all_posts = []
    for sub in subreddits[:3]:   # max 3 subs per symbol
        posts = fetch_reddit_posts(sub, query=symbol, limit=limit)
        if not posts:
            # If query filter returns nothing, grab hot posts
            posts = fetch_reddit_posts(sub, limit=10)
        all_posts.extend(posts)
    return all_posts[:40]
