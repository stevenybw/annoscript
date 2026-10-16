"""Fenced metric persistence. No provider bodies or client-side journal."""

import asyncio
import time
import uuid

from lev.job_protocol import TERMINAL, decode, encode
from lev.metrics import aggregate, normalize_usage


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
            if self._owned(db, claim) is None:
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
    def __init__(self, store, claim, backend, *, clock=time.monotonic):
        self.store, self.claim, self.backend, self.clock = store, claim, backend, clock

    async def start(self, questions):
        identity = uuid.uuid4().hex
        if not await asyncio.to_thread(self.store.request_started, self.claim, identity, self.backend, questions):
            raise asyncio.CancelledError()
        return identity

    async def finish(self, identity, elapsed, status, usage):
        pending = asyncio.create_task(asyncio.to_thread(self.store.request_finished, self.claim, identity,
                                                        seconds=elapsed, http_status=status, usage=usage))
        try:
            committed = await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Runtime can cancel speculative requests after another call fails.
            # Finish an already observed HTTP outcome before the attempt ends;
            # the store still rejects revoked ownership. Never detach a writer.
            await pending
            raise
        if not committed:
            raise asyncio.CancelledError()
