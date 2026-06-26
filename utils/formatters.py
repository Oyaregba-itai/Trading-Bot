def fmt_price(price: float | None, decimals: int = 2) -> str:
    if price is None:
        return "N/A"
    if price < 0.001:
        return f"${price:.8f}"
    if price < 1:
        return f"${price:.6f}"
    if price < 100:
        return f"${price:.4f}"
    return f"${price:,.{decimals}f}"


def fmt_change(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    arrow = "▲" if pct >= 0 else "▼"
    return f"{arrow} {abs(pct):.2f}%"


def fmt_large(n: float | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n / 1_000:.2f}K"
    return f"${n:.2f}"


def signal_emoji(signal: str) -> str:
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "NEUTRAL": "🟡 NEUTRAL"}.get(signal, signal)


def rsi_bar(val: float) -> str:
    filled = int(val / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {val:.1f}"


def format_crypto(data: dict) -> str:
    change_24 = fmt_change(data.get("change_24h"))
    change_7d = fmt_change(data.get("change_7d"))
    lines = [
        f"*{data['name']} ({data['symbol']})*",
        f"Price: {fmt_price(data['price'])}",
        f"24h: {change_24}  |  7d: {change_7d}",
        f"High/Low: {fmt_price(data['high_24h'])} / {fmt_price(data['low_24h'])}",
        f"Market Cap: {fmt_large(data.get('market_cap'))}",
        f"Volume: {fmt_large(data.get('volume_24h'))}",
    ]
    if data.get("rank"):
        lines.append(f"Rank: #{data['rank']}")
    if data.get("ath"):
        lines.append(f"ATH: {fmt_price(data['ath'])} ({fmt_change(data.get('ath_change'))})")
    return "\n".join(lines)


def format_stock(data: dict) -> str:
    lines = [
        f"*{data.get('name', data['symbol'])} ({data['symbol']})*",
        f"Price: {fmt_price(data['price'])}",
        f"Change: {fmt_change(data.get('change_24h'))}",
        f"High/Low: {fmt_price(data.get('high'))} / {fmt_price(data.get('low'))}",
        f"Volume: {fmt_large(data.get('volume'))}",
    ]
    if data.get("market_cap"):
        lines.append(f"Market Cap: {fmt_large(data['market_cap'])}")
    return "\n".join(lines)


def format_forex(data: dict) -> str:
    lines = [
        f"*{data['pair']}*",
        f"Price: {data['price']:.5f}",
    ]
    if data.get("bid") and data.get("ask"):
        lines.append(f"Bid: {data['bid']:.5f}  |  Ask: {data['ask']:.5f}")
    if data.get("change_24h") is not None:
        lines.append(f"Change: {fmt_change(data['change_24h'])}")
    return "\n".join(lines)


def format_analysis(symbol: str, price: float, analysis: dict) -> str:
    lines = [f"*Technical Analysis: {symbol}*", f"Price: {fmt_price(price)}", ""]

    if analysis.get("rsi"):
        lines.append(f"RSI(14): {rsi_bar(analysis['rsi'])}")
        lines.append(f"  → {analysis.get('rsi_signal', '')}")

    if analysis.get("macd"):
        m = analysis["macd"]
        lines.append(f"\nMACD: {m['macd']:+.4f}  Signal: {m['signal']:+.4f}  Hist: {m['histogram']:+.4f}")
        lines.append(f"  → {analysis.get('macd_signal', '')}")

    if analysis.get("sma_20") and analysis.get("sma_50"):
        lines.append(f"\nSMA20: {fmt_price(analysis['sma_20'])}  SMA50: {fmt_price(analysis['sma_50'])}")
        trend = "Uptrend (SMA20 > SMA50)" if analysis["sma_20"] > analysis["sma_50"] else "Downtrend (SMA20 < SMA50)"
        lines.append(f"  → {trend}")

    if analysis.get("bb"):
        bb = analysis["bb"]
        lines.append(f"\nBollinger Bands:")
        lines.append(f"  Upper: {fmt_price(bb['upper'])}  Mid: {fmt_price(bb['middle'])}  Lower: {fmt_price(bb['lower'])}")

    lines.append(f"\n*Overall Signal: {signal_emoji(analysis.get('overall_signal', 'NEUTRAL'))}*")
    return "\n".join(lines)


def format_news(articles: list[dict], title: str = "Latest News") -> str:
    if not articles:
        return "No news found."
    lines = [f"*{title}*", ""]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. [{a['title']}]({a['url']})")
        lines.append(f"   _{a.get('source', '')}_ • {a.get('published', '')}")
        lines.append("")
    return "\n".join(lines)
