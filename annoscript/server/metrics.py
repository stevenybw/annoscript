"""Fenced metric persistence. No provider bodies or client-side journal."""

import asyncio
import time
import uuid

from annoscript.job_protocol import TERMINAL, decode, encode
from annoscript.metrics import aggregate, normalize_usage
from annoscript.backends.base import BackendFatalError


UPGRADE = """
ALTER TABLE jobs ADD COLUMN metrics_available INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN queued_monotonic REAL;
ALTER TABLE jobs ADD COLUMN clock_epoch TEXT;
ALTER TABLE attempts ADD COLUMN metered INTEGER NOT NULL DEFAULT 0;
ALTER TABLE attempts ADD COLUMN duration_seconds REAL;
ALTER TABLE attempts ADD COLUMN queue_wait_seconds REAL;
CREATE TABLE http_requests (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, idx INTEGER NOT NULL, attempt INTEGER NOT NULL,
 backend TEXT NOT NULL, questions INTEGER NOT NULL, http_status INTEGER, seconds REAL, usage BLOB,
 FOREIGN KEY(job_id,idx,attempt) REFERENCES attempts(job_id,idx,number));
CREATE INDEX http_requests_job ON http_requests(job_id,idx,attempt);
PRAGMA user_version=2;
"""


class MetricStore:
    def meter_attempt(self, claim, supported):
        with self._tx() as db:
            if self._owned(db, claim) is None:
                return False
            db.execute("UPDATE attempts SET metered=? WHERE job_id=? AND idx=? AND number=?",
                       (int(supported), claim["job_id"], claim["index"], claim["attempt"]))
            return True

    def request_started(self, claim, request_id, backend, questions):
        if backend not in ("decision", "generation") or type(questions) is not int or questions < 1:
            raise ValueError("invalid request metrics")
        with self._tx() as db:
            owned = self._owned(db, claim)
            if owned is None or owned[0]["state"] not in ("queued", "running"):
                return False
            args = (request_id, claim["job_id"], claim["index"], claim["attempt"], backend, questions)
            old = db.execute("SELECT id,job_id,idx,attempt,backend,questions FROM http_requests WHERE id=?", (request_id,)).fetchone()
            if old is not None and tuple(old) != args:
                raise ValueError("request metric identity conflict")
            db.execute("INSERT OR IGNORE INTO http_requests(id,job_id,idx,attempt,backend,questions) VALUES (?,?,?,?,?,?)", args)
            return True

    def request_finished(self, claim, request_id, *, seconds, http_status, usage):
        if type(seconds) not in (int, float) or not 0 <= seconds < float("inf"):
            raise ValueError("invalid HTTP duration")
        if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
            raise ValueError("invalid HTTP status")
        with self._tx() as db:
            if self._owned(db, claim) is None:
                return False
            db.execute("""UPDATE http_requests SET seconds=?,http_status=?,usage=?
                          WHERE id=? AND job_id=? AND idx=? AND attempt=? AND seconds IS NULL""",
                       (seconds, http_status, encode(normalize_usage(usage)), request_id,
                        claim["job_id"], claim["index"], claim["attempt"]))
            return True

    def _attempt_duration(self, db, claim, seconds):
        if seconds is not None and (type(seconds) not in (int, float) or not 0 <= seconds < float("inf")):
            raise ValueError("invalid attempt duration")
        db.execute("UPDATE attempts SET duration_seconds=? WHERE job_id=? AND idx=? AND number=?",
                   (seconds, claim["job_id"], claim["index"], claim["attempt"]))

    def metrics(self, principal, job_id):
        with self._tx() as db:
            job = self._job(db, job_id, principal)
            status = self._status(db, job)
            attempts = [dict(r) for r in db.execute("SELECT * FROM attempts WHERE job_id=? ORDER BY idx,number", (job_id,))]
            requests = [dict(r) for r in db.execute("SELECT * FROM http_requests WHERE job_id=? ORDER BY id", (job_id,))]
            for row in requests:
                row["usage"] = decode(row["usage"]) if row["usage"] is not None else normalize_usage(None)
            items, waits = [], []
            for index in status["input_indices"]:
                aa = [a for a in attempts if a["idx"] == index]
                rr = [r for r in requests if r["idx"] == index]
                # Queue intervals are valid only within the same monotonic
                # clock epoch (boot); never subtract wall timestamps.
                wait = aa[0]["queue_wait_seconds"] if aa else None
                waits.append(wait)
                items.append({"index": index, "metrics": aggregate(aa, rr, history=bool(job["metrics_available"]), waits=[wait])})
            return {"protocol_version": 1, "metrics_version": 1, "job": status,
                    "provisional": status["state"] not in TERMINAL,
                    "summary": aggregate(attempts, requests, history=bool(job["metrics_available"]), waits=waits), "inputs": items}


class RequestMeter:
    """One meter per owned attempt, invoked only at the actual HTTP boundary."""
    def __init__(self, store, claim, backend, *, clock=time.monotonic, observer=None, wall_clock=time.time):
        self.store, self.claim, self.backend, self.clock = store, claim, backend, clock
        self.observer, self.wall_clock = observer, wall_clock

    def _observe(self, event):
        if self.observer is not None:
            try:
                self.observer({"job_id": self.claim["job_id"], "index": self.claim["index"],
                               "attempt": self.claim["attempt"], "stage": self.backend, **event})
            except Exception:
                raise BackendFatalError("request audit persistence failed", code="metrics_failed") from None

    async def start(self, questions):
        identity = uuid.uuid4().hex
        try:
            committed = await asyncio.to_thread(self.store.request_started, self.claim, identity, self.backend, questions)
        except Exception:
            raise BackendFatalError("request intent persistence failed", code="metrics_failed") from None
        if not committed:
            if self.store.renew(self.claim):
                raise BackendFatalError("job is paused; no new model request was sent", code="job_paused")
            raise asyncio.CancelledError()
        self._observe({"event": "intent", "request_id": identity, "questions": questions,
                       "started_unix": self.wall_clock()})
        return identity

    async def finish(self, identity, elapsed, status, usage):
        outcome = {"event": "outcome", "request_id": identity, "seconds": elapsed,
                   "http_status": status, "usage": usage, "finished_unix": self.wall_clock()}
        committed = False
        pending = asyncio.create_task(asyncio.to_thread(self.store.request_finished, self.claim, identity,
                                                        seconds=elapsed, http_status=status, usage=usage))
        try:
            try:
                committed = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Drain persistence and audit before propagating cancellation.
                # Ownership fencing still applies to stale attempts.
                committed = await pending
                raise
        except Exception:
            raise BackendFatalError("request outcome persistence failed", code="metrics_failed") from None
        finally:
            self._observe({**outcome, "storage_committed": committed})
        if not committed:
            raise asyncio.CancelledError()
