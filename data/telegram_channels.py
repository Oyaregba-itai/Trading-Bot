"""
Scrape public Telegram channel previews (no API key needed).
Uses t.me/s/CHANNEL_NAME — the public web preview of a channel.
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Well-known public crypto/trading Telegram channels
PUBLIC_CHANNELS = [
    "bitcoin",
    "CryptoPanic",
    "binance",
    "CryptoComOfficial",
    "coingecko",
    "CoinMarketCap",
    "cryptosignals",
    "tradingviewcom",
]

CRYPTO_SIGNAL_CHANNELS = [
    "crypto_trading_signals",
    "cryptosignals",
    "bitcoinwisdom",
    "whale_alert_io",
]


def scrape_channel(channel_name: str, count: int = 10) -> list[dict]:
    url = f"https://t.me/s/{channel_name}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        messages = soup.find_all("div", class_="tgme_widget_message_text")
        results = []
        for msg in messages[-count:]:
            text = msg.get_text(separator=" ", strip=True)
            if text and len(text) > 10:
                results.append({"text": text[:400], "channel": channel_name})
        return results
    except Exception:
        return []


def fetch_telegram_sentiment_data(symbol: str, count_per_channel: int = 5) -> list[dict]:
    """Collect messages from public Telegram channels mentioning the symbol."""
    results = []
    sym_lower = symbol.lower()

    for channel in PUBLIC_CHANNELS + CRYPTO_SIGNAL_CHANNELS:
        messages = scrape_channel(channel, count=20)
        for msg in messages:
            if sym_lower in msg["text"].lower() or any(
                kw in msg["text"].lower() for kw in ["crypto", "bitcoin", "market", "signal", "buy", "sell"]
            ):
                results.append(msg)
                if len(results) >= count_per_channel * len(PUBLIC_CHANNELS):
                    return results

    return results


def fetch_whale_alerts() -> list[dict]:
    """Whale Alert public channel for large transaction signals."""
    return scrape_channel("whale_alert_io", count=10)
