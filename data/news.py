import requests
import feedparser
from config import NEWS_API_KEY, NEWS_API_BASE


def get_news_newsapi(query: str = "cryptocurrency trading", count: int = 5) -> list[dict]:
    if not NEWS_API_KEY:
        return []
    try:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": count,
            "language": "en",
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(f"{NEWS_API_BASE}/everything", params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "url": a["url"],
                "published": a["publishedAt"][:10],
            }
            for a in articles
        ]
    except Exception:
        return []


# Free RSS feeds — no API key needed
RSS_FEEDS = {
    "crypto": "https://cointelegraph.com/rss",
    "crypto2": "https://cryptonews.com/news/feed",
    "stocks": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "forex": "https://www.forexlive.com/feed",
    "general": "https://feeds.reuters.com/reuters/businessNews",
}


def get_rss_news(category: str = "crypto", count: int = 5) -> list[dict]:
    url = RSS_FEEDS.get(category.lower(), RSS_FEEDS["crypto"])
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:count]:
            articles.append({
                "title": entry.get("title", "No title"),
                "source": feed.feed.get("title", category),
                "url": entry.get("link", ""),
                "published": entry.get("published", "")[:10] if entry.get("published") else "",
            })
        return articles
    except Exception:
        return []


def get_trading_news(symbol: str = "", count: int = 5) -> list[dict]:
    """Get news for a specific symbol or general trading news."""
    if symbol:
        # Try NewsAPI first with symbol query
        news = get_news_newsapi(f"{symbol} trading price", count)
        if news:
            return news
    # Fall back to RSS
    category = "crypto" if _is_crypto_symbol(symbol) else "general"
    return get_rss_news(category, count)


def _is_crypto_symbol(symbol: str) -> bool:
    from config import CRYPTO_IDS
    return symbol.upper() in CRYPTO_IDS
