"""
Binance Testnet broker integration.
Places real paper orders on Binance Testnet instead of the internal demo wallet.
Stop-loss and take-profit orders execute instantly at the exchange level.

Testnet URL: https://testnet.binance.vision
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TESTNET_BASE_URL = "https://testnet.binance.vision"

# Binance symbol map: our symbol -> Binance trading pair
_SYMBOL_MAP = {
    "BTC":   "BTCUSDT",
    "ETH":   "ETHUSDT",
    "SOL":   "SOLUSDT",
    "BNB":   "BNBUSDT",
    "XRP":   "XRPUSDT",
    "ADA":   "ADAUSDT",
    "AVAX":  "AVAXUSDT",
    "DOT":   "DOTUSDT",
    "MATIC": "MATICUSDT",
    "LINK":  "LINKUSDT",
    "LTC":   "LTCUSDT",
    "TRX":   "TRXUSDT",
    "TON":   "TONUSDT",
    "NEAR":  "NEARUSDT",
    "SUI":   "SUIUSDT",
    "ATOM":  "ATOMUSDT",
    "DOGE":  "DOGEUSDT",
    "PEPE":  "PEPEUSDT",
    "WIF":   "WIFUSDT",
    "TURBO": "TURBOUSDT",
}

_CRYPTO_SYMBOLS = set(_SYMBOL_MAP.keys())


def _get_client():
    """Create a Binance testnet client."""
    try:
        from binance.client import Client
        api_key    = os.environ.get("BINANCE_TESTNET_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
        if not api_key or not api_secret:
            return None
        client = Client(api_key, api_secret, testnet=True)
        return client
    except ImportError:
        logger.error("python-binance not installed. Run: pip install python-binance")
        return None
    except Exception as e:
        logger.error("Binance testnet client error: %s", e)
        return None


def is_available() -> bool:
    """Check if Binance Testnet is configured and reachable."""
    client = _get_client()
    if not client:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO_SYMBOLS


def get_binance_symbol(symbol: str) -> Optional[str]:
    return _SYMBOL_MAP.get(symbol.upper())


def get_account_balance() -> float:
    """Get USDT balance from Binance Testnet account."""
    client = _get_client()
    if not client:
        return 0.0
    try:
        account = client.get_account()
        for asset in account["balances"]:
            if asset["asset"] == "USDT":
                return float(asset["free"])
        return 0.0
    except Exception as e:
        logger.error("Failed to get Binance balance: %s", e)
        return 0.0


def get_symbol_info(binance_symbol: str) -> dict:
    """Get lot size and price filter for a symbol."""
    client = _get_client()
    if not client:
        return {}
    try:
        info = client.get_symbol_info(binance_symbol)
        result = {}
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                result["min_qty"]  = float(f["minQty"])
                result["step_qty"] = float(f["stepSize"])
            if f["filterType"] == "PRICE_FILTER":
                result["tick_size"] = float(f["tickSize"])
        return result
    except Exception as e:
        logger.error("Failed to get symbol info for %s: %s", binance_symbol, e)
        return {}


def _round_step(qty: float, step: float) -> float:
    """Round quantity to valid lot step size."""
    if step <= 0:
        return qty
    import math
    precision = max(0, round(-math.log10(step)))
    return round(round(qty / step) * step, precision)


def place_buy_order(symbol: str, usdt_amount: float, stop_loss_price: float, take_profit_price: float) -> Optional[dict]:
    """
    Place a market buy order on Binance Testnet with OCO stop-loss/take-profit.
    Returns order info dict or None on failure.
    """
    client = _get_client()
    if not client:
        return None

    binance_sym = get_binance_symbol(symbol)
    if not binance_sym:
        logger.warning("No Binance symbol mapping for %s", symbol)
        return None

    try:
        # Get current price
        ticker = client.get_symbol_ticker(symbol=binance_sym)
        price  = float(ticker["price"])

        # Get lot size rules
        info     = get_symbol_info(binance_sym)
        step_qty = info.get("step_qty", 0.00001)
        min_qty  = info.get("min_qty", 0.00001)

        # Calculate quantity
        raw_qty = usdt_amount / price
        qty     = _round_step(raw_qty, step_qty)
        if qty < min_qty:
            logger.warning("Quantity %.8f below min %.8f for %s", qty, min_qty, symbol)
            return None

        # Place market buy
        buy_order = client.order_market_buy(symbol=binance_sym, quantity=qty)
        logger.info("Binance BUY: %s qty=%.8f @ ~$%.4f", symbol, qty, price)

        # Place OCO (stop-loss + take-profit) sell order
        tick = info.get("tick_size", 0.01)
        sl_price = _round_step(stop_loss_price, tick)
        tp_price = _round_step(take_profit_price, tick)

        # OCO requires stop_price slightly above limit for stop-loss side
        sl_limit = _round_step(sl_price * 0.999, tick)

        try:
            oco_order = client.order_oco_sell(
                symbol=binance_sym,
                quantity=qty,
                price=str(tp_price),          # take profit limit
                stopPrice=str(sl_price),       # stop loss trigger
                stopLimitPrice=str(sl_limit),  # stop loss limit
                stopLimitTimeInForce="GTC",
            )
            logger.info("Binance OCO placed for %s: SL=%.4f TP=%.4f", symbol, sl_price, tp_price)
        except Exception as e:
            logger.warning("OCO order failed for %s (will use bot monitoring): %s", symbol, e)
            oco_order = None

        return {
            "symbol":       symbol,
            "binance_sym":  binance_sym,
            "price":        price,
            "quantity":     qty,
            "cost":         qty * price,
            "stop_loss":    sl_price,
            "take_profit":  tp_price,
            "buy_order_id": buy_order.get("orderId"),
            "oco_order":    oco_order,
            "source":       "binance_testnet",
        }

    except Exception as e:
        logger.error("Binance buy order failed for %s: %s", symbol, e)
        return None


def place_sell_order(symbol: str, quantity: float) -> Optional[dict]:
    """Place a market sell order to close a position."""
    client = _get_client()
    if not client:
        return None

    binance_sym = get_binance_symbol(symbol)
    if not binance_sym:
        return None

    try:
        # Cancel any existing OCO orders first
        try:
            open_orders = client.get_open_orders(symbol=binance_sym)
            for order in open_orders:
                client.cancel_order(symbol=binance_sym, orderId=order["orderId"])
        except Exception:
            pass

        # Get lot size and round quantity
        info     = get_symbol_info(binance_sym)
        step_qty = info.get("step_qty", 0.00001)
        qty      = _round_step(quantity, step_qty)

        order = client.order_market_sell(symbol=binance_sym, quantity=qty)

        ticker = client.get_symbol_ticker(symbol=binance_sym)
        price  = float(ticker["price"])

        logger.info("Binance SELL: %s qty=%.8f @ ~$%.4f", symbol, qty, price)
        return {
            "symbol":    symbol,
            "quantity":  qty,
            "price":     price,
            "order_id":  order.get("orderId"),
            "source":    "binance_testnet",
        }

    except Exception as e:
        logger.error("Binance sell order failed for %s: %s", symbol, e)
        return None


def get_open_positions() -> list:
    """Get all non-zero crypto balances from Binance Testnet."""
    client = _get_client()
    if not client:
        return []
    try:
        account  = client.get_account()
        usdt_map = {v: k for k, v in _SYMBOL_MAP.items()}
        positions = []
        for bal in account["balances"]:
            asset = bal["asset"]
            qty   = float(bal["free"]) + float(bal["locked"])
            if asset == "USDT" or qty < 0.000001:
                continue
            sym_pair = asset + "USDT"
            our_sym  = usdt_map.get(sym_pair)
            if our_sym:
                positions.append({"symbol": our_sym, "quantity": qty})
        return positions
    except Exception as e:
        logger.error("Failed to get Binance positions: %s", e)
        return []
