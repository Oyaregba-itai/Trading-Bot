"""
Expanded RSS feeds: Bloomberg, Reuters, CNBC, CoinDesk, CoinTelegraph,
Decrypt, The Block, MarketWatch, ForexLive, FXStreet, Investing.com.
Filters by symbol relevance and scores with VADER.
"""
import feedparser

# Feed definitions grouped by category
_RSS_FEEDS = {
    "crypto": [
        ("CoinDesk",        "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph",   "https://cointelegraph.com/rss"),
        ("Decrypt",         "https://decrypt.co/feed"),
        ("The Block",       "https://www.theblock.co/rss.xml"),
        ("CryptoSlate",     "https://cryptoslate.com/feed/"),
        ("BeInCrypto",      "https://beincrypto.com/feed/"),
        ("NewsBTC",         "https://www.newsbtc.com/feed/"),
        ("Bitcoin Magazine","https://bitcoinmagazine.com/feed"),
    ],
    "stocks": [
        ("MarketWatch",     "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("Seeking Alpha",   "https://seekingalpha.com/feed.xml"),
        ("Investopedia",    "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline"),
        ("Motley Fool",     "https://www.fool.com/a/feeds/foolwatch?format=rss2"),
        ("Benzinga",        "https://www.benzinga.com/feed"),
    ],
    "forex": [
        ("ForexLive",       "https://www.forexlive.com/feed/"),
        ("FXStreet",        "https://www.fxstreet.com/rss/news"),
        ("DailyFX",         "https://www.dailyfx.com/feeds/all"),
        ("Investing.com",   "https://www.investing.com/rss/news.rss"),
    ],
    "general": [
        ("Reuters",         "https://feeds.reuters.com/reuters/businessNews"),
        ("CNBC",            "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Bloomberg",       "https://feeds.bloomberg.com/markets/news.rss"),
        ("Financial Times", "https://www.ft.com/?format=rss"),
        ("WSJ Markets",     "https://feeds.content.dowjones.io/public/rss/mw_marketpulse"),
    ],
}

# Map symbol → category to pick the best feeds
_CRYPTO_SYMS = {"BTC", "ETH", "SOL", "BNB", "DOGE", "PEPE", "WIF", "BONK",
                "FLOKI", "XRP", "ADA", "AVAX", "MATIC", "LINK"}
_FOREX_SYMS  = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
                "NZDUSD", "EURGBP", "EURJPY"}
_COMMODITY_SYMS = {"GOLD", "SILVER", "OIL", "NATGAS"}


def _pick_categories(symbol: str) -> list[str]:
    sym = symbol.upper()
    if sym in _CRYPTO_SYMS:
        return ["crypto", "general"]
    if sym in _FOREX_SYMS or sym in _COMMODITY_SYMS:
        return ["forex", "general"]
    return ["stocks", "general"]


def fetch_rss_headlines(symbol: str, max_per_feed: int = 5, timeout: int = 8) -> list[str]:
    """
    Fetch and filter headlines from all relevant RSS feeds.
    Returns headlines that mention the symbol (or all recent ones if too few).
    """
    sym = symbol.upper()
    categories = _pick_categories(sym)
    feeds = []
    for cat in categories:
        feeds.extend(_RSS_FEEDS.get(cat, []))

    import threading
    all_headlines: list[str] = []
    lock = threading.Lock()

    def _fetch_one(name: str, url: str):
        try:
            parsed = feedparser.parse(url)
            titles = [e.get("title", "") for e in parsed.entries[:max_per_feed] if e.get("title")]
            with lock:
                all_headlines.extend(titles)
        except Exception:
            pass

    threads = [threading.Thread(target=_fetch_one, args=(n, u)) for n, u in feeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)

    # Filter: prefer headlines that mention the symbol, fall back to all
    sym_headlines = [h for h in all_headlines
                     if sym.lower() in h.lower()
                     or symbol.lower() in h.lower()]

    if len(sym_headlines) >= 5:
        return sym_headlines[:40]

    # Also search for common name aliases
    aliases = _get_aliases(symbol)
    alias_headlines = [h for h in all_headlines
                       if any(a in h.lower() for a in aliases)]
    combined = list(dict.fromkeys(sym_headlines + alias_headlines))  # deduplicate
    return combined[:40] if combined else all_headlines[:20]


def _get_aliases(symbol: str) -> list[str]:
    alias_map = {
        "BTC":    ["bitcoin", "btc"],
        "ETH":    ["ethereum", "ether", "eth"],
        "SOL":    ["solana", "sol"],
        "BNB":    ["binance", "bnb"],
        "DOGE":   ["dogecoin", "doge"],
        "PEPE":   ["pepe", "meme coin", "memecoin"],
        "XRP":    ["ripple", "xrp"],
        "ADA":    ["cardano", "ada"],
        "AAPL":   ["apple", "aapl"],
        "TSLA":   ["tesla", "tsla", "elon musk"],
        "EURUSD": ["euro", "eur/usd", "eurusd", "ecb", "federal reserve"],
        "GOLD":   ["gold", "xau", "precious metal"],
        "OIL":    ["oil", "crude", "wti", "brent", "opec"],
    }
    return alias_map.get(symbol.upper(), [symbol.lower()])
