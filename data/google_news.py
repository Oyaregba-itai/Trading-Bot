"""Google News RSS — no API key needed."""
import feedparser
import requests
from urllib.parse import quote_plus


def fetch_google_news(query: str, count: int = 10) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:count]:
            results.append({
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", "Google News"),
                "url": entry.get("link", ""),
                "published": entry.get("published", "")[:10],
                "summary": entry.get("summary", ""),
            })
        return results
    except Exception:
        return []


def fetch_symbol_news(symbol: str, count: int = 15) -> list[dict]:
    queries = [
        f"{symbol} cryptocurrency price",
        f"{symbol} trading signal",
        f"{symbol} crypto news",
    ]
    seen = set()
    results = []
    for q in queries:
        for item in fetch_google_news(q, count // len(queries) + 2):
            if item["title"] not in seen:
                seen.add(item["title"])
                results.append(item)
    return results[:count]
