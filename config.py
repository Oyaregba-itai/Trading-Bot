import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
NEWS_API_BASE = "https://newsapi.org/v2"

ALERT_CHECK_INTERVAL = 60

# ── Crypto (all fetchable via yfinance SYMBOL-USD) ────────────────────────────
CRYPTO_IDS = {
    # Large caps
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "BNB":   "binancecoin",
    "SOL":   "solana",
    "XRP":   "ripple",
    "ADA":   "cardano",
    "AVAX":  "avalanche-2",
    "DOT":   "polkadot",
    "MATIC": "matic-network",
    "LINK":  "chainlink",
    "LTC":   "litecoin",
    "BCH":   "bitcoin-cash",
    "TRX":   "tron",
    "NEAR":  "near",
    "APT":   "aptos",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "UNI":   "uniswap",
    "TON":   "the-open-network",
    "SUI":   "sui",
    "SEI":   "sei-network",
    "INJ":   "injective-protocol",
    "FET":   "fetch-ai",
    "RENDER":"render-token",
    "ATOM":  "cosmos",
    "XLM":   "stellar",
    "VET":   "vechain",
    "ALGO":  "algorand",
    "HBAR":  "hedera-hashgraph",
    "ICP":   "internet-computer",
    # Meme coins
    "DOGE":  "dogecoin",
    "SHIB":  "shiba-inu",
    "PEPE":  "pepe",
    "WIF":   "dogwifcoin",
    "BONK":  "bonk",
    "FLOKI": "floki",
    "BRETT": "brett",
    "MOG":   "mog-coin",
    "TURBO": "turbo",
    "BABYDOGE": "baby-doge-coin",
    "NEIRO": "neiro",
    "POPCAT":"popcat",
    # Stablecoins (tracked for sentiment/dominance, not price-traded)
    "USDT":  "tether",
    "USDC":  "usd-coin",
    "DAI":   "dai",
    "BUSD":  "binance-usd",
}

# Stablecoins — skip auto-trading (price pegged to $1, no movement to predict)
STABLECOIN_SYMBOLS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE"}

# Meme coins
MEME_COIN_SYMBOLS = {
    "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI",
    "BRETT", "MOG", "TURBO", "BABYDOGE", "NEIRO", "POPCAT",
}

# ── Forex ─────────────────────────────────────────────────────────────────────
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY",
]

# ── Stocks & ETFs ─────────────────────────────────────────────────────────────
STOCK_SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL",
    "NVDA", "META", "SPY", "QQQ", "AMD",
    "NFLX", "PYPL", "INTC", "DIS", "BABA",
]

# ── Commodities ───────────────────────────────────────────────────────────────
COMMODITY_SYMBOLS = {
    "GOLD":   "GC=F",
    "SILVER": "SI=F",
    "OIL":    "CL=F",
    "NATGAS": "NG=F",
    "WHEAT":  "ZW=F",
    "COPPER": "HG=F",
}
