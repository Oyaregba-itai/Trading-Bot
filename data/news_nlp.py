"""
Real-time news NLP using free RSS feeds + VADER sentiment.
Detects high-impact events: Fed statements, earnings beats/misses, macro surprises.
No API key required — all sources are public RSS.
"""
import time
import logging
import feedparser
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Per-symbol cache: symbol → {score, impact, headline, ts}
_cache: dict[str, dict] = {}
_CACHE_TTL = 600   # 10 minutes

# ── RSS feed sources ──────────────────────────────────────────────────────────

_GENERAL_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.bloomberg.com/markets/news.rss",
]
_CRYPTO_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopanic.com/news/rss/",
]
_FOREX_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.forexfactory.com/rss",
]

# ── Impact keyword classifier ─────────────────────────────────────────────────

_HIGH_IMPACT = {
    # Fed / Central banks
    "federal reserve", "fed rate", "rate hike", "rate cut", "fomc",
    "interest rate", "powell", "lagarde", "boe", "ecb",
    # Macro data
    "nonfarm payroll", "nfp", "cpi", "inflation", "pce", "gdp", "recession",
    "unemployment", "jobs report", "retail sales",
    # Crisis keywords
    "bankruptcy", "default", "collapse", "bailout", "contagion", "crisis",
    # Crypto specific
    "sec", "etf approval", "halving", "hack", "exploit", "ban", "seized",
    # Earnings
    "earnings beat", "earnings miss", "guidance cut", "guidance raised",
    "revenue beat", "revenue miss", "profit warning",
}

_BULLISH_WORDS = {
    "beat", "beats", "surpassed", "exceeded", "raised", "upgrade",
    "record high", "all-time high", "approval", "partnership", "launch",
    "growth", "strong", "positive", "bullish", "rally", "soared",
}
_BEARISH_WORDS = {
    "miss", "missed", "fell short", "cut", "downgrade", "layoffs",
    "warning", "decline", "drop", "crash", "ban", "hack", "exploit",
    "default", "bankruptcy", "bearish", "plunge", "tumble",
}


def _vader_score(text: str) -> float:
    """VADER compound sentiment in [-1, +1]. Downloads lexicon once if needed."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def _keyword_score(text: str) -> tuple[float, bool]:
    """
    Returns (directional_score, is_high_impact).
    directional_score > 0 = bullish keywords dominate, < 0 = bearish.
    """
    lower = text.lower()
    is_high = any(kw in lower for kw in _HIGH_IMPACT)
    bull = sum(1 for w in _BULLISH_WORDS if w in lower)
    bear = sum(1 for w in _BEARISH_WORDS if w in lower)
    total = bull + bear
    if total == 0:
        return 0.0, is_high
    return round((bull - bear) / total, 4), is_high


def _fetch_feed(url: str, max_age_hours: int = 4) -> list[str]:
    """Fetch RSS and return list of recent headlines."""
    try:
        feed = feedparser.parse(url)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        headlines = []
        for entry in feed.entries[:20]:
            # Parse published date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                import calendar
                ts = calendar.timegm(entry.published_parsed)
                published = datetime.fromtimestamp(ts, tz=timezone.utc)
            if published and published < cutoff:
                continue
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            headlines.append(f"{title}. {summary}"[:300])
        return headlines
    except Exception as e:
        logger.debug("RSS feed error %s: %s", url, e)
        return []


def _is_relevant(text: str, symbol: str) -> bool:
    """Check if headline mentions the symbol or its name."""
    lower = text.lower()
    sym = symbol.lower()
    # Direct symbol mention
    if sym in lower:
        return True
    # Common name mappings
    names = {
        "btc": ["bitcoin"], "eth": ["ethereum"], "sol": ["solana"],
        "bnb": ["binance"], "doge": ["dogecoin"], "avax": ["avalanche"],
        "xrp": ["ripple"], "ada": ["cardano"], "dot": ["polkadot"],
        "eurusd": ["euro", "eur/usd", "eurusd"],
        "gbpusd": ["sterling", "pound", "gbp/usd"],
        "usdjpy": ["yen", "usd/jpy"], "usdcad": ["cad", "loonie"],
        "gold": ["gold", "xau"], "oil": ["crude", "wti", "brent"],
        "aapl": ["apple"], "tsla": ["tesla"], "nvda": ["nvidia"],
        "msft": ["microsoft"], "googl": ["google", "alphabet"],
        "meta": ["facebook", "meta platform"], "amzn": ["amazon"],
    }
    for alias in names.get(sym, []):
        if alias in lower:
            return True
    return False


# ── Main public function ──────────────────────────────────────────────────────

def get_news_signal(symbol: str) -> dict:
    """
    Fetch and score recent news for a symbol.
    Returns:
      sentiment    : float [-1, +1]  — VADER + keyword combined
      impact       : bool            — True if high-impact event detected
      top_headline : str             — most relevant recent headline
      n_articles   : int             — number of articles analysed
    """
    neutral = {"sentiment": 0.0, "impact": False, "top_headline": "", "n_articles": 0}

    cached = _cache.get(symbol.upper(), {})
    if cached and (time.time() - cached.get("_ts", 0)) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    from config import CRYPTO_IDS
    try:
        feeds = _CRYPTO_FEEDS if symbol in CRYPTO_IDS else _GENERAL_FEEDS
        all_headlines: list[str] = []
        for url in feeds:
            all_headlines.extend(_fetch_feed(url))

        relevant = [h for h in all_headlines if _is_relevant(h, symbol)]
        if not relevant:
            # Fall back to general macro headlines
            relevant = all_headlines[:5]

        if not relevant:
            return neutral

        scores, impacts, top = [], [], relevant[0]
        for h in relevant[:10]:
            v_score = _vader_score(h)
            k_score, is_high = _keyword_score(h)
            combined = v_score * 0.6 + k_score * 0.4
            scores.append(combined)
            if is_high:
                impacts.append(h)

        avg_score = round(sum(scores) / len(scores), 4)
        is_high_impact = len(impacts) > 0
        if impacts:
            top = impacts[0]

        result = {
            "sentiment":     max(-1.0, min(1.0, avg_score)),
            "impact":        is_high_impact,
            "top_headline":  top[:200],
            "n_articles":    len(relevant),
        }
        _cache[symbol.upper()] = {**result, "_ts": time.time()}
        return result

    except Exception as e:
        logger.debug("News NLP error for %s: %s", symbol, e)
        return neutral
