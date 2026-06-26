"""
YouTube data via channel RSS feeds (no API key) + transcript extraction.
Falls back gracefully if transcripts are unavailable.
"""
import feedparser
import requests
from xml.etree import ElementTree

# Popular crypto/trading YouTube channels {name: channel_id}
CRYPTO_CHANNELS = {
    "CoinBureau":     "UCqK_GSMbpiV8spgD3ZGloSw",
    "BenjaminCowen":  "UCRvqjQPSeaWn-uEx-w0XOIg",
    "AltcoinDaily":   "UCbLhGKVY-bJPcawebgtNfbw",
    "DataDash":       "UCCatR7nWbYrkVXdxXb4cGXtQ",
    "InvestAnswers":  "UCnKdnNomPKEqKHHRqbKRhng",
    "MilesDestiny":   "UCpEFgktfkJ0R7bP_o-oaMcw",
    "CryptoRUs":      "UCRjfSBMbsqBNYMJPRpvVJZg",
}

STOCK_CHANNELS = {
    "Graham Stephan": "UCV6KDgJskWaEckne5aPA0aQ",
    "Andrei Jikh":    "UCGy7SkBjcIAgTiwkXEtPnYg",
    "TickerSymbol:U": "UCbmNph6atAoGfqLoCL_duAg",
}


def get_channel_videos(channel_id: str, count: int = 5) -> list[dict]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        feed = feedparser.parse(url)
        videos = []
        for entry in feed.entries[:count]:
            video_id = entry.get("yt_videoid", "")
            if not video_id:
                link = entry.get("link", "")
                video_id = link.split("v=")[-1] if "v=" in link else ""
            videos.append({
                "title": entry.get("title", ""),
                "video_id": video_id,
                "url": entry.get("link", ""),
                "published": entry.get("published", "")[:10],
                "channel": feed.feed.get("title", channel_id),
            })
        return videos
    except Exception:
        return []


def get_transcript(video_id: str, max_chars: int = 1000) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(seg["text"] for seg in transcript)
        return text[:max_chars]
    except Exception:
        return ""


def fetch_youtube_sentiment_data(symbol: str, max_videos: int = 8) -> list[dict]:
    """Fetch recent video titles (and transcripts if available) about a symbol."""
    results = []
    seen = set()

    channels = {**CRYPTO_CHANNELS, **STOCK_CHANNELS}
    for name, cid in channels.items():
        for video in get_channel_videos(cid, count=3):
            title_lower = video["title"].lower()
            sym_lower = symbol.lower()
            # Only include videos mentioning the symbol
            if sym_lower in title_lower or any(
                kw in title_lower for kw in ["crypto", "bitcoin", "altcoin", "market"]
            ):
                if video["title"] not in seen:
                    seen.add(video["title"])
                    # Try to get transcript for extra context
                    if video["video_id"]:
                        video["transcript"] = get_transcript(video["video_id"], 500)
                    else:
                        video["transcript"] = ""
                    results.append(video)

        if len(results) >= max_videos:
            break

    return results[:max_videos]


def search_youtube_rss(query: str) -> list[dict]:
    """Search YouTube via Google News RSS for YouTube videos."""
    try:
        from urllib.parse import quote_plus
        url = f"https://news.google.com/rss/search?q={quote_plus(query + ' site:youtube.com')}&hl=en-US"
        feed = feedparser.parse(url)
        return [{"title": e.get("title", ""), "url": e.get("link", "")} for e in feed.entries[:5]]
    except Exception:
        return []
