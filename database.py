import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")
INITIAL_BALANCE = 10_000.0


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS wallet (
            id          INTEGER PRIMARY KEY,
            cash        REAL    NOT NULL,
            initial     REAL    NOT NULL,
            updated_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS open_positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL UNIQUE,
            asset_type  TEXT    NOT NULL,
            entry_price REAL    NOT NULL,
            quantity    REAL    NOT NULL,
            cost        REAL    NOT NULL,
            stop_loss   REAL,
            take_profit REAL,
            confidence  REAL,
            opened_at   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            asset_type  TEXT    NOT NULL,
            entry_price REAL    NOT NULL,
            exit_price  REAL,
            quantity    REAL    NOT NULL,
            cost        REAL    NOT NULL,
            revenue     REAL,
            pnl         REAL,
            pnl_pct     REAL,
            result      TEXT,
            confidence  REAL,
            signal      TEXT,
            opened_at   TEXT    NOT NULL,
            closed_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS model_registry (
            symbol      TEXT    PRIMARY KEY,
            accuracy    REAL,
            precision_s REAL,
            recall_s    REAL,
            f1_s        REAL,
            n_samples   INTEGER,
            trained_at  TEXT    NOT NULL,
            model_path  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            signal      TEXT    NOT NULL,
            confidence  REAL,
            price       REAL,
            sentiment   REAL,
            predicted_at TEXT   NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sentiment_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            score       REAL    NOT NULL,
            headline    TEXT,
            logged_at   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS autotrade_sessions (
            chat_id     INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            PRIMARY KEY (chat_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS grid_bots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            low         REAL    NOT NULL,
            high        REAL    NOT NULL,
            levels      INTEGER NOT NULL,
            grid_size   REAL    NOT NULL,
            total_profit REAL   DEFAULT 0,
            trades_done  INTEGER DEFAULT 0,
            last_price  REAL,
            state       TEXT    NOT NULL DEFAULT '{}',
            created_at  TEXT    NOT NULL,
            active      INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS dca_schedules (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id         INTEGER NOT NULL,
            symbol          TEXT    NOT NULL,
            amount_usd      REAL    NOT NULL,
            interval_type   TEXT    NOT NULL,
            next_run        TEXT    NOT NULL,
            total_invested  REAL    DEFAULT 0,
            total_qty       REAL    DEFAULT 0,
            avg_price       REAL    DEFAULT 0,
            runs_done       INTEGER DEFAULT 0,
            active          INTEGER DEFAULT 1,
            created_at      TEXT    NOT NULL
        );
    """)

    # Seed wallet if empty
    c.execute("SELECT COUNT(*) FROM wallet")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO wallet (id, cash, initial, updated_at) VALUES (1, ?, ?, ?)",
            (INITIAL_BALANCE, INITIAL_BALANCE, datetime.utcnow().isoformat())
        )

    conn.commit()
    conn.close()


# ── Wallet helpers ────────────────────────────────────────────────────────────

def get_cash() -> float:
    with get_conn() as c:
        row = c.execute("SELECT cash FROM wallet WHERE id=1").fetchone()
        return row["cash"] if row else INITIAL_BALANCE


def get_initial() -> float:
    with get_conn() as c:
        row = c.execute("SELECT initial FROM wallet WHERE id=1").fetchone()
        return row["initial"] if row else INITIAL_BALANCE


def set_cash(amount: float):
    with get_conn() as c:
        c.execute("UPDATE wallet SET cash=?, updated_at=? WHERE id=1",
                  (amount, datetime.utcnow().isoformat()))


def reset_wallet():
    with get_conn() as c:
        c.execute("UPDATE wallet SET cash=?, initial=?, updated_at=? WHERE id=1",
                  (INITIAL_BALANCE, INITIAL_BALANCE, datetime.utcnow().isoformat()))
        c.execute("DELETE FROM open_positions")


# ── Position helpers ──────────────────────────────────────────────────────────

def get_position(symbol: str) -> sqlite3.Row | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM open_positions WHERE symbol=?", (symbol,)).fetchone()


def get_all_positions() -> list:
    with get_conn() as c:
        return c.execute("SELECT * FROM open_positions").fetchall()


def open_position(symbol, asset_type, entry_price, quantity, cost, stop_loss, take_profit, confidence):
    with get_conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO open_positions
            (symbol, asset_type, entry_price, quantity, cost, stop_loss, take_profit, confidence, opened_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (symbol, asset_type, entry_price, quantity, cost, stop_loss, take_profit, confidence,
              datetime.utcnow().isoformat()))


def update_stop_loss(symbol: str, new_stop: float):
    with get_conn() as c:
        c.execute("UPDATE open_positions SET stop_loss=? WHERE symbol=?", (new_stop, symbol))


def close_position(symbol: str):
    with get_conn() as c:
        c.execute("DELETE FROM open_positions WHERE symbol=?", (symbol,))


# ── Trade history helpers ─────────────────────────────────────────────────────

def record_trade(symbol, asset_type, entry_price, exit_price, quantity, cost, revenue,
                 pnl, pnl_pct, result, confidence, signal, opened_at):
    with get_conn() as c:
        c.execute("""
            INSERT INTO trade_history
            (symbol, asset_type, entry_price, exit_price, quantity, cost, revenue,
             pnl, pnl_pct, result, confidence, signal, opened_at, closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (symbol, asset_type, entry_price, exit_price, quantity, cost, revenue,
              pnl, pnl_pct, result, confidence, signal, opened_at,
              datetime.utcnow().isoformat()))


def get_trade_history(limit: int = 20) -> list:
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM trade_history ORDER BY closed_at DESC LIMIT ?", (limit,)
        ).fetchall()


def get_all_closed_trades() -> list:
    with get_conn() as c:
        return c.execute("SELECT * FROM trade_history WHERE exit_price IS NOT NULL").fetchall()


# ── Model registry helpers ────────────────────────────────────────────────────

def save_model_meta(symbol, accuracy, precision_s, recall_s, f1_s, n_samples, model_path):
    with get_conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO model_registry
            (symbol, accuracy, precision_s, recall_s, f1_s, n_samples, trained_at, model_path)
            VALUES (?,?,?,?,?,?,?,?)
        """, (symbol, accuracy, precision_s, recall_s, f1_s, n_samples,
              datetime.utcnow().isoformat(), model_path))


def get_model_meta(symbol: str) -> sqlite3.Row | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM model_registry WHERE symbol=?", (symbol,)).fetchone()


def get_all_models() -> list:
    with get_conn() as c:
        return c.execute("SELECT * FROM model_registry ORDER BY trained_at DESC").fetchall()


# ── Prediction log helpers ────────────────────────────────────────────────────

def log_prediction(symbol, signal, confidence, price, sentiment):
    with get_conn() as c:
        c.execute("""
            INSERT INTO prediction_log (symbol, signal, confidence, price, sentiment, predicted_at)
            VALUES (?,?,?,?,?,?)
        """, (symbol, signal, confidence, price, sentiment, datetime.utcnow().isoformat()))


# ── Autotrade session persistence ────────────────────────────────────────────

def save_last_chat_id(chat_id: int):
    with get_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_config (key TEXT PRIMARY KEY, value TEXT);
        """)
        c.execute("INSERT OR REPLACE INTO bot_config (key,value) VALUES ('last_chat_id',?)",
                  (str(chat_id),))


def get_last_chat_id() -> int | None:
    try:
        with get_conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS bot_config (key TEXT PRIMARY KEY, value TEXT)")
            row = c.execute("SELECT value FROM bot_config WHERE key='last_chat_id'").fetchone()
            return int(row["value"]) if row else None
    except Exception:
        return None


def save_autotrade_session(chat_id: int, symbols: set):
    save_last_chat_id(chat_id)
    with get_conn() as c:
        c.execute("DELETE FROM autotrade_sessions WHERE chat_id=?", (chat_id,))
        for sym in symbols:
            c.execute("INSERT OR IGNORE INTO autotrade_sessions (chat_id, symbol) VALUES (?,?)",
                      (chat_id, sym))


def delete_autotrade_session(chat_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM autotrade_sessions WHERE chat_id=?", (chat_id,))


def load_all_autotrade_sessions() -> dict:
    """Returns {chat_id: set(symbols)} for all saved sessions."""
    with get_conn() as c:
        rows = c.execute("SELECT chat_id, symbol FROM autotrade_sessions").fetchall()
    result: dict[int, set] = {}
    for row in rows:
        result.setdefault(row["chat_id"], set()).add(row["symbol"])
    return result


def log_sentiment(symbol, source, score, headline=""):
    with get_conn() as c:
        c.execute("""
            INSERT INTO sentiment_log (symbol, source, score, headline, logged_at)
            VALUES (?,?,?,?,?)
        """, (symbol, source, score, headline, datetime.utcnow().isoformat()))


# ── Grid bot helpers ──────────────────────────────────────────────────────────

def create_grid(chat_id, symbol, low, high, levels, grid_size, state_json):
    with get_conn() as c:
        c.execute("""
            INSERT INTO grid_bots (chat_id, symbol, low, high, levels, grid_size, state, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (chat_id, symbol, low, high, levels, grid_size, state_json, datetime.utcnow().isoformat()))
        return c.lastrowid

def get_active_grids(chat_id=None):
    with get_conn() as c:
        if chat_id:
            return c.execute("SELECT * FROM grid_bots WHERE active=1 AND chat_id=?", (chat_id,)).fetchall()
        return c.execute("SELECT * FROM grid_bots WHERE active=1").fetchall()

def update_grid(grid_id, last_price, state_json, profit_delta=0, trades_delta=0):
    with get_conn() as c:
        c.execute("""
            UPDATE grid_bots SET last_price=?, state=?,
              total_profit=total_profit+?, trades_done=trades_done+?
            WHERE id=?
        """, (last_price, state_json, profit_delta, trades_delta, grid_id))

def stop_grid(grid_id):
    with get_conn() as c:
        c.execute("UPDATE grid_bots SET active=0 WHERE id=?", (grid_id,))


# ── DCA helpers ───────────────────────────────────────────────────────────────

def create_dca(chat_id, symbol, amount_usd, interval_type, next_run):
    with get_conn() as c:
        c.execute("""
            INSERT INTO dca_schedules (chat_id, symbol, amount_usd, interval_type, next_run, created_at)
            VALUES (?,?,?,?,?,?)
        """, (chat_id, symbol, amount_usd, interval_type, next_run, datetime.utcnow().isoformat()))
        return c.lastrowid

def get_active_dcas(chat_id=None):
    with get_conn() as c:
        if chat_id:
            return c.execute("SELECT * FROM dca_schedules WHERE active=1 AND chat_id=?", (chat_id,)).fetchall()
        return c.execute("SELECT * FROM dca_schedules WHERE active=1").fetchall()

def update_dca(dca_id, next_run, qty_bought, amount_spent, new_avg):
    with get_conn() as c:
        c.execute("""
            UPDATE dca_schedules SET next_run=?, total_qty=total_qty+?,
              total_invested=total_invested+?, avg_price=?, runs_done=runs_done+1
            WHERE id=?
        """, (next_run, qty_bought, amount_spent, new_avg, dca_id))

def stop_dca(dca_id):
    with get_conn() as c:
        c.execute("UPDATE dca_schedules SET active=0 WHERE id=?", (dca_id,))
