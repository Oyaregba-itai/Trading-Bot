import requests
import yfinance as yf
from config import COINGECKO_BASE, CRYPTO_IDS


def get_crypto_price(symbol: str) -> dict | None:
    symbol = symbol.upper()
    coin_id = CRYPTO_IDS.get(symbol)

    # If not in our map, try searching CoinGecko
    if not coin_id:
        coin_id = _search_coin(symbol)
        if not coin_id:
            return None

    try:
        url = f"{COINGECKO_BASE}/coins/{coin_id}"
        params = {"localization": "false", "tickers": "false", "community_data": "false"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        md = d["market_data"]
        return {
            "symbol": symbol,
            "name": d["name"],
            "price": md["current_price"]["usd"],
            "change_24h": md["price_change_percentage_24h"],
            "change_7d": md["price_change_percentage_7d"],
            "market_cap": md["market_cap"]["usd"],
            "volume_24h": md["total_volume"]["usd"],
            "high_24h": md["high_24h"]["usd"],
            "low_24h": md["low_24h"]["usd"],
            "ath": md["ath"]["usd"],
            "ath_change": md["ath_change_percentage"]["usd"],
            "rank": d["market_cap_rank"],
        }
    except Exception:
        return None


def _search_coin(query: str) -> str | None:
    try:
        r = requests.get(f"{COINGECKO_BASE}/search", params={"query": query}, timeout=10)
        r.raise_for_status()
        coins = r.json().get("coins", [])
        if coins:
            return coins[0]["id"]
    except Exception:
        pass
    return None


def get_top_coins(limit: int = 10) -> list[dict]:
    try:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
        r = requests.get(f"{COINGECKO_BASE}/coins/markets", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_trending_coins() -> list[dict]:
    try:
        r = requests.get(f"{COINGECKO_BASE}/search/trending", timeout=10)
        r.raise_for_status()
        return r.json().get("coins", [])
    except Exception:
        return []


def get_crypto_history(symbol: str, days: int = 30) -> list:
    """Returns list of [timestamp, price] for charting/indicators."""
    coin_id = CRYPTO_IDS.get(symbol.upper()) or _search_coin(symbol)
    if not coin_id:
        return []
    try:
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        r = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("prices", [])
    except Exception:
        return []


def get_fear_greed() -> dict | None:
    try:
        from config import FEAR_GREED_URL
        r = requests.get(FEAR_GREED_URL, timeout=10)
        r.raise_for_status()
        data = r.json()["data"][0]
        return {"value": int(data["value"]), "classification": data["value_classification"]}
    except Exception:
        return None
