"""
NewsAPI.org — returns up to 100 headlines per query (free tier: 100 req/day).
Only runs if NEWS_API_KEY is set in .env.
Falls back silently to empty if key is missing.
"""
import os
import requests

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def fetch_newsapi(symbol: str, limit: int = 20) -> list[str]:
    if not NEWS_API_KEY:
        return []

    _name_map = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binance coin", "DOGE": "dogecoin", "PEPE": "pepe coin",
        "AAPL": "apple stock", "TSLA": "tesla stock",
        "EURUSD": "euro dollar", "GOLD": "gold price", "OIL": "crude oil price",
    }
    query = _name_map.get(symbol.upper(), symbol)

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        articles = r.json().get("articles", [])
        return [f"{a.get('title','')} {a.get('description','')}" for a in articles if a.get("title")]
    except Exception:
        return []
