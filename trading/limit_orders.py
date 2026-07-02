"""
Limit order simulation for the paper wallet.
Instead of buying at market price, we place a limit buy slightly below market
and wait up to 30 minutes for the price to come to us — better fills, less slippage.

Flow:
  1. execute_buy() → place_limit_buy() if limit orders enabled
  2. Position monitor (every 30s) → check_limit_fills()
  3. If price ≤ limit_price → fill the order → create open position
  4. If expires_at passed → cancel and log
"""
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Limit price offset per asset class (how much below market to place the limit)
_LIMIT_OFFSET = {
    "crypto":    0.0010,   # 0.10% below market (bid-ask spread is tight)
    "forex":     0.0003,   # 0.03% below market
    "commodity": 0.0005,   # 0.05% below market
    "stock":     0.0010,   # 0.10% below market
}

# How long limit orders stay open before expiring
_EXPIRY_MINUTES = {
    "5m": 15,    # scalp: expire fast
    "1h": 60,    # intraday: 1 hour window
    "1d": 240,   # swing: 4 hour window
}


def place_limit_buy(symbol: str, asset_type: str, market_price: float,
                    quantity: float, cost: float, stop_loss: float,
                    take_profit: float, confidence: float,
                    timeframe: str, signal: str) -> dict:
    """
    Place a limit buy order below the current market price.
    Returns the pending order dict.
    """
    import database as db
    from trading.ws_price_feed import get_cached_spread

    offset = _LIMIT_OFFSET.get(asset_type, 0.001)

    # Use real bid price from WebSocket for crypto if available
    spread = get_cached_spread(symbol) if asset_type == "crypto" else None
    if spread:
        limit_price = spread[0]   # bid price (already below ask/mid)
    else:
        limit_price = market_price * (1 - offset)

    expiry_mins = _EXPIRY_MINUTES.get(timeframe, 60)
    expires_at  = (datetime.now(timezone.utc) + timedelta(minutes=expiry_mins)).isoformat()

    db.place_pending_order(
        symbol=symbol, asset_type=asset_type,
        limit_price=round(limit_price, 8), quantity=quantity, cost=cost,
        stop_loss=stop_loss, take_profit=take_profit,
        confidence=confidence, timeframe=timeframe, signal=signal,
        expires_at=expires_at,
    )

    saving_pct = (market_price - limit_price) / market_price * 100
    logger.info("Limit buy placed: %s @ %.6f (%.3f%% below market, expires %s)",
                symbol, limit_price, saving_pct, expires_at[:16])
    return {
        "symbol":      symbol,
        "limit_price": limit_price,
        "quantity":    quantity,
        "cost":        cost,
        "expires_at":  expires_at,
        "type":        "LIMIT_BUY",
    }


def check_limit_fills(get_price_fn) -> list[dict]:
    """
    Called by the position monitor every 30 seconds.
    Checks all pending limit orders against current prices.
    Returns list of filled orders (each becomes an open position).
    """
    import database as db

    now     = datetime.now(timezone.utc)
    orders  = db.get_all_pending_orders()
    filled  = []
    expired = []

    for order in orders:
        order = dict(order)
        symbol      = order["symbol"]
        limit_price = order["limit_price"]
        expires_at  = order["expires_at"]

        # Check expiry
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if now > exp:
                expired.append(symbol)
                db.delete_pending_order(symbol)
                logger.info("Limit order expired: %s @ %.6f", symbol, limit_price)
                continue
        except Exception:
            pass

        # Check fill
        current_price = get_price_fn(symbol)
        if current_price is None:
            continue

        if current_price <= limit_price:
            # Filled! Create the open position
            try:
                db.open_position(
                    symbol      = symbol,
                    asset_type  = order["asset_type"],
                    entry_price = current_price,     # fill at current price (may be better than limit)
                    quantity    = order["quantity"],
                    cost        = order["cost"],
                    stop_loss   = order["stop_loss"],
                    take_profit = order["take_profit"],
                    confidence  = order["confidence"],
                )
                db.delete_pending_order(symbol)
                saving = (limit_price - current_price) / limit_price * 100
                logger.info("Limit order filled: %s @ %.6f (%.4f%% better than limit)",
                            symbol, current_price, saving)
                filled.append({
                    "symbol":      symbol,
                    "fill_price":  current_price,
                    "limit_price": limit_price,
                    "quantity":    order["quantity"],
                    "cost":        order["cost"],
                    "stop_loss":   order["stop_loss"],
                    "take_profit": order["take_profit"],
                    "timeframe":   order.get("timeframe", "1h"),
                    "type":        "LIMIT_FILL",
                })
            except Exception as e:
                logger.error("Limit fill error %s: %s", symbol, e)

    return filled
