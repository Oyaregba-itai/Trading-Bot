"""
Aggregate sentiment from 12 sources concurrently:
  1.  Google News RSS
  2.  Reddit crypto/stock/forex subreddits (via RSS — JSON API blocked in 2023)
  3.  YouTube channel RSS + transcripts
  4.  Telegram public channels
  5.  CryptoPanic API
  6.  Extra Reddit subreddits (WSB, r/investing, r/algotrading, etc.)
  7.  Finviz news table (US stocks)
  8.  Expanded RSS (Bloomberg, Reuters, CNBC, CoinDesk, CoinTelegraph, Decrypt, etc.)
  9.  Yahoo Finance RSS
 10.  Hacker News (Algolia API)
 11.  Legacy RSS feeds
 12.  NewsAPI (only if NEWS_API_KEY is set in .env)
"""
import threading
import requests

def _ensure_vader():
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass

_ensure_vader()


def _vader_score(text: str) -> float:
    if not text or len(text) < 5:
        return 0.0
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    except Exception:
        return _keyword_score(text)


def _keyword_score(text: str) -> float:
    t = text.lower()
    bullish = ["bull","pump","moon","ath","buy","long","surge","rally","breakout",
               "growth","gain","profit","rise","higher","strong","adoption"]
    bearish = ["bear","dump","crash","sell","short","drop","fall","loss","down",
               "lower","weak","fear","panic","rug","scam","correction","plunge"]
    b = sum(1 for w in bullish if w in t)
    s = sum(1 for w in bearish if w in t)
    total = b + s
    return 0.0 if total == 0 else (b - s) / total


def _cryptopanic_news(symbol: str) -> list[str]:
    try:
        r = requests.get("https://cryptopanic.com/api/v1/posts/",
                         params={"auth_token": "anonymous", "public": "true", "currencies": symbol},
                         timeout=8)
        r.raise_for_status()
        return [item.get("title","") for item in r.json().get("results",[])[:10]]
    except Exception:
        return []


def _hacker_news(symbol: str, limit: int = 10) -> list[str]:
    _names = {"BTC":"bitcoin","ETH":"ethereum","SOL":"solana","BNB":"binance",
              "AAPL":"apple","TSLA":"tesla","GOLD":"gold price","OIL":"oil price"}
    query = _names.get(symbol.upper(), symbol)
    try:
        r = requests.get(
            f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage={limit}",
            timeout=8)
        return [h.get("title","") for h in r.json().get("hits",[]) if h.get("title")]
    except Exception:
        return []


class SentimentResult:
    def __init__(self):
        self.scores: list[float] = []
        self.sources: dict[str, list] = {}

    def add(self, source: str, texts: list[str]):
        scores = [_vader_score(t) for t in texts if t and len(t) > 3]
        self.sources[source] = scores
        self.scores.extend(scores)

    @property
    def composite(self) -> float:
        return round(sum(self.scores) / len(self.scores), 4) if self.scores else 0.0

    @property
    def breakdown(self) -> dict:
        out = {}
        for src, scores in self.sources.items():
            if scores:
                out[src] = {
                    "avg":      round(sum(scores) / len(scores), 3),
                    "count":    len(scores),
                    "positive": sum(1 for s in scores if s > 0.05),
                    "negative": sum(1 for s in scores if s < -0.05),
                    "neutral":  sum(1 for s in scores if -0.05 <= s <= 0.05),
                }
        return out


def aggregate_sentiment(symbol: str, log_to_db: bool = True) -> SentimentResult:
    result  = SentimentResult()
    data: dict[str, list[str]] = {}

    def _google():
        try:
            from data.google_news import fetch_symbol_news
            items = fetch_symbol_news(symbol, count=15)
            data["google_news"] = [f"{i['title']} {i.get('summary','')}" for i in items]
        except Exception:
            data["google_news"] = []

    def _reddit():
        try:
            from data.reddit_data import fetch_symbol_reddit
            posts = fetch_symbol_reddit(symbol, limit=20)
            data["reddit"] = [f"{p['title']} {p.get('text','')}" for p in posts]
        except Exception:
            data["reddit"] = []

    def _youtube():
        try:
            from data.youtube_data import fetch_youtube_sentiment_data
            videos = fetch_youtube_sentiment_data(symbol, max_videos=8)
            texts = []
            for v in videos:
                texts.append(v["title"])
                if v.get("transcript"):
                    texts.append(v["transcript"][:300])
            data["youtube"] = texts
        except Exception:
            data["youtube"] = []

    def _telegram():
        try:
            from data.telegram_channels import fetch_telegram_sentiment_data
            data["telegram"] = [m["text"] for m in fetch_telegram_sentiment_data(symbol)]
        except Exception:
            data["telegram"] = []

    def _cryptopanic():
        data["cryptopanic"] = _cryptopanic_news(symbol)

    def _extra_reddit():
        try:
            from data.stocktwits import fetch_extra_reddit
            data["reddit_extra"] = fetch_extra_reddit(symbol, limit=20)
        except Exception:
            data["reddit_extra"] = []

    def _finviz():
        try:
            from data.finviz import fetch_finviz_news
            data["finviz"] = fetch_finviz_news(symbol, limit=15)
        except Exception:
            data["finviz"] = []

    def _web_rss():
        try:
            from data.web_rss import fetch_rss_headlines
            data["web_rss"] = fetch_rss_headlines(symbol, max_per_feed=5)
        except Exception:
            data["web_rss"] = []

    def _yahoo_finance():
        try:
            from data.stocktwits import fetch_yahoo_finance_news
            data["yahoo_finance"] = fetch_yahoo_finance_news(symbol, limit=10)
        except Exception:
            data["yahoo_finance"] = []

    def _hackernews():
        data["hacker_news"] = _hacker_news(symbol)

    def _legacy_rss():
        try:
            from data.news import get_rss_news
            articles = get_rss_news("crypto", 8)
            texts = [a["title"] for a in articles if symbol.lower() in a["title"].lower()]
            data["rss_feeds"] = texts or [a["title"] for a in articles[:5]]
        except Exception:
            data["rss_feeds"] = []

    def _newsapi():
        try:
            from data.newsapi import fetch_newsapi
            data["newsapi"] = fetch_newsapi(symbol, limit=20)
        except Exception:
            data["newsapi"] = []

    collectors = [_google, _reddit, _youtube, _telegram, _cryptopanic,
                  _extra_reddit, _finviz, _web_rss, _yahoo_finance,
                  _hackernews, _legacy_rss, _newsapi]

    workers = [threading.Thread(target=fn) for fn in collectors]
    for w in workers: w.start()
    for w in workers: w.join(timeout=20)

    for source, texts in data.items():
        result.add(source, texts)

    if log_to_db:
        try:
            from database import log_sentiment
            for source, texts in data.items():
                for text in texts[:3]:
                    log_sentiment(symbol, source, _vader_score(text), text[:200])
        except Exception:
            pass

    return result


def get_sentiment_label(score: float) -> str:
    if score >= 0.3:  return "Very Bullish"
    if score >= 0.1:  return "Bullish"
    if score >= -0.1: return "Neutral"
    if score >= -0.3: return "Bearish"
    return "Very Bearish"


def sentiment_emoji(score: float) -> str:
    if score >= 0.3:  return "🟢🟢"
    if score >= 0.1:  return "🟢"
    if score >= -0.1: return "🟡"
    if score >= -0.3: return "🔴"
    return "🔴🔴"
