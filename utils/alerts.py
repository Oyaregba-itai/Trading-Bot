"""
In-memory alert store. Alerts survive as long as the bot process runs.
Each alert: {id, chat_id, symbol, asset_type, condition, target, created}
"""
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Alert:
    alert_id: str
    chat_id: int
    symbol: str
    asset_type: str        # "crypto" | "stock" | "forex" | "commodity"
    condition: str         # "above" | "below"
    target: float
    created: float = field(default_factory=time.time)
    triggered: bool = False


_alerts: dict[str, Alert] = {}


def add_alert(chat_id: int, symbol: str, asset_type: str, condition: str, target: float) -> Alert:
    aid = str(uuid.uuid4())[:8]
    alert = Alert(aid, chat_id, symbol.upper(), asset_type, condition, target)
    _alerts[aid] = alert
    return alert


def get_user_alerts(chat_id: int) -> list[Alert]:
    return [a for a in _alerts.values() if a.chat_id == chat_id and not a.triggered]


def remove_alert(alert_id: str, chat_id: int) -> bool:
    alert = _alerts.get(alert_id)
    if alert and alert.chat_id == chat_id:
        del _alerts[alert_id]
        return True
    return False


def get_all_active_alerts() -> list[Alert]:
    return [a for a in _alerts.values() if not a.triggered]


def mark_triggered(alert_id: str):
    if alert_id in _alerts:
        _alerts[alert_id].triggered = True


def check_alert(alert: Alert, current_price: float) -> bool:
    if alert.condition == "above":
        return current_price >= alert.target
    elif alert.condition == "below":
        return current_price <= alert.target
    return False
