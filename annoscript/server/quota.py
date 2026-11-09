"""Persistent admission counters; no user inputs or model responses are stored."""
from datetime import datetime
import hashlib
import hmac
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo

class Quota:
    """Persist anonymous admission counts, never inputs or model results."""
    def __init__(self, path, secret, *, daily=1000, ip_daily=50, minute=5):
        self.path, self.secret = str(path), secret.encode()
        self.daily, self.ip_daily, self.minute = daily, ip_daily, minute
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS daily(day TEXT, ip TEXT, count INTEGER,
                    PRIMARY KEY(day, ip));
                CREATE TABLE IF NOT EXISTS admissions(ip TEXT, at REAL);
                CREATE INDEX IF NOT EXISTS admissions_ip_at ON admissions(ip, at);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def admit(self, ip, now=None):
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now, ZoneInfo("Asia/Shanghai")).date().isoformat()
        identity = hmac.new(self.secret, ip.encode(), hashlib.sha256).hexdigest()
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM admissions WHERE at <= ?", (now - 60,))
            db.execute("DELETE FROM daily WHERE day < ?", (day,))
            total = db.execute("SELECT COALESCE(SUM(count),0) FROM daily WHERE day=?", (day,)).fetchone()[0]
            own = db.execute("SELECT count FROM daily WHERE day=? AND ip=?", (day, identity)).fetchone()
            recent = db.execute("SELECT MIN(at),COUNT(*) FROM admissions WHERE ip=?", (identity,)).fetchone()
            if total >= self.daily:
                return "daily_limit", 3600
            if own and own[0] >= self.ip_daily:
                return "ip_daily_limit", 3600
            if recent[1] >= self.minute:
                return "rate_limit", max(1, int(recent[0] + 61 - now))
            db.execute("INSERT INTO daily VALUES(?,?,1) ON CONFLICT(day,ip) DO UPDATE SET count=count+1", (day, identity))
            db.execute("INSERT INTO admissions VALUES(?,?)", (identity, now))
            db.commit()
            return None
        finally:
            db.close()

