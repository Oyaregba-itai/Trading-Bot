"""
Finviz news scraper — US stocks only (AAPL, TSLA, NVDA, etc.).
Scrapes the news table from finviz.com/quote.ashx.
No API key needed.
"""
import requests
from bs4 import BeautifulSoup


def fetch_finviz_news(symbol: str, limit: int = 15) -> list[str]:
    """
    Returns list of news headlines for a US stock symbol.
    Returns [] for crypto/forex — Finviz only covers equities.
    """
    # Only useful for stock tickers (skip crypto and forex)
    _non_stock = {
        "BTC", "ETH", "SOL", "BNB", "DOGE", "PEPE", "WIF", "BONK", "FLOKI",
        "EURUSD", "GBPUSD", "USDJPY", "GOLD", "SILVER", "OIL", "NATGAS",
    }
    if symbol.upper() in _non_stock:
        return []

    try:
        url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://finviz.com/",
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "lxml")
        # News table has class "fullview-news-outer" or rows with id "news-table"
        table = soup.find("table", {"id": "news-table"})
        if not table:
            return []

        headlines = []
        for row in table.find_all("tr")[:limit]:
            td = row.find_all("td")
            if len(td) >= 2:
                link = td[1].find("a")
                if link:
                    headlines.append(link.get_text(strip=True))

        return headlines
    except Exception:
        return []
