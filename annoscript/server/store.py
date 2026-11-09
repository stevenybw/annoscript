"""Transactional job journal. All model work happens outside these transactions."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
import uuid

from annoscript.job_protocol import (TERMINAL, cursor_for, cursor_seq, decode, encode,
                              request_sha256, validate_request)
from annoscript.server.metrics import MetricStore, UPGRADE


class ServiceError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


SCHEMA_V1 = """
CREATE TABLE releases (id TEXT PRIMARY KEY, manifest BLOB NOT NULL);
CREATE TABLE sessions (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, idem_key TEXT NOT NULL,
 state TEXT NOT NULL, expires REAL NOT NULL, created REAL NOT NULL,
 UNIQUE(principal, idem_key));
CREATE TABLE jobs (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, idem_key TEXT NOT NULL,
 request_hash TEXT NOT NULL, request BLOB, ir BLOB, digest TEXT NOT NULL,
 mode TEXT NOT NULL, release_id TEXT NOT NULL REFERENCES releases(id),
 session_id TEXT REFERENCES sessions(id), state TEXT NOT NULL, indices BLOB,
 pause_generation INTEGER NOT NULL DEFAULT 0, pause_reason BLOB,
 seq INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL,
 terminal_at REAL, expired INTEGER NOT NULL DEFAULT 0,
 UNIQUE(principal, idem_key));
CREATE TABLE inputs (
 job_id TEXT NOT NULL REFERENCES jobs(id), idx INTEGER NOT NULL, state BLOB NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending', round_attempts INTEGER NOT NULL DEFAULT 0,
 total_attempts INTEGER NOT NULL DEFAULT 0, token INTEGER NOT NULL DEFAULT 0,
 lease_until REAL, next_attempt REAL NOT NULL DEFAULT 0,
 PRIMARY KEY(job_id, idx));
CREATE TABLE attempts (
 job_id TEXT NOT NULL, idx INTEGER NOT NULL, number INTEGER NOT NULL,
 token INTEGER NOT NULL, worker TEXT NOT NULL, status TEXT NOT NULL,
 started REAL NOT NULL, finished REAL, error BLOB, metadata BLOB,
 PRIMARY KEY(job_id, idx, number),
 FOREIGN KEY(job_id, idx) REFERENCES inputs(job_id, idx));
CREATE TABLE results (
 job_id TEXT NOT NULL, idx INTEGER NOT NULL, seq INTEGER NOT NULL,
 status TEXT NOT NULL, value BLOB NOT NULL, error BLOB NOT NULL, attempts INTEGER NOT NULL,
 PRIMARY KEY(job_id, idx), UNIQUE(job_id, seq),
 FOREIGN KEY(job_id, idx) REFERENCES inputs(job_id, idx));
CREATE TABLE resume_ops (
 job_id TEXT NOT NULL REFERENCES jobs(id), idem_key TEXT NOT NULL,
 generation INTEGER NOT NULL, PRIMARY KEY(job_id, idem_key));
CREATE INDEX inputs_ready ON inputs(status, next_attempt);
CREATE INDEX jobs_sessions ON jobs(session_id, state);
PRAGMA user_version=1;
"""


SCHEMA = SCHEMA_V1 + UPGRADE


class Store(MetricStore):
    def __init__(self, path, *, clock=time.time, session_ttl=300.0, lease_seconds=60.0,
                 retention_seconds=7 * 86400, monotonic=time.monotonic):
        self.path, self.clock, self.monotonic = str(path), clock, monotonic
        try:
            self.clock_epoch = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            self.clock_epoch = None
        self.session_ttl, self.lease_seconds, self.retention_seconds = session_ttl, lease_seconds, retention_seconds
        with self._connection() as db:
            # Switching journal mode can return BUSY immediately when several
            # processes initialize the same file. Retry only this startup step.
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    db.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as error:
                    code = getattr(error, "sqlite_errorcode", 0)
                    if code & 0xff != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                # Never treat an unversioned existing database as a fresh one.
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                    raise RuntimeError("unversioned nonempty job database")
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        db.execute(statement)
            elif version != 2:
                raise RuntimeError(f"unsupported job database schema {version}")
            db.commit()

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _tx(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._maintenance(db)
                yield db
            except ServiceError:
                # Maintenance remains durable even if the requested operation
                # is rejected. Operations validate before changing their data.
                db.commit()
                raise
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    def maintain(self):
        with self._tx():
            pass

    def _job(self, db, job_id, principal=None):
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or (principal is not None and job["principal"] != principal):
            raise ServiceError("not_found", "job not found", 404)
        if job["expired"]:
            raise ServiceError("job_expired", "job data has expired", 410)
        return job

    def _session(self, db, session_id, principal):
        row = db.execute("SELECT * FROM sessions WHERE id=? AND principal=?", (session_id, principal)).fetchone()
        if row is None:
            raise ServiceError("not_found", "session not found", 404)
        return row

    def _session_value(self, row):
        return {"session_id": row["id"], "state": row["state"], "expires_at": row["expires"],
                "heartbeat_interval": min(30.0, self.session_ttl / 3)}

    def create_session(self, principal, idem_key):
        with self._tx() as db:
            row = db.execute("SELECT * FROM sessions WHERE principal=? AND idem_key=?", (principal, idem_key)).fetchone()
            if row is None:
                sid, now = uuid.uuid4().hex, self.clock()
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", (sid, principal, idem_key, "active", now + self.session_ttl, now))
                row = self._session(db, sid, principal)
            return self._session_value(row)

    def heartbeat_session(self, principal, session_id):
        with self._tx() as db:
            row = self._session(db, session_id, principal)
            if row["state"] != "active":
                raise ServiceError("session_closed", "session is closed or expired", 409)
            db.execute("UPDATE sessions SET expires=? WHERE id=?", (self.clock() + self.session_ttl, session_id))
            return self._session_value(self._session(db, session_id, principal))

    def close_session(self, principal, session_id):
        with self._tx() as db:
            row = self._session(db, session_id, principal)
            if row["state"] == "active":
                db.execute("UPDATE sessions SET state='closed' WHERE id=?", (session_id,))
                self._cancel_session_jobs(db, session_id, "client_closed")
            return self._session_value(self._session(db, session_id, principal))

    def _cancel_session_jobs(self, db, session_id, code):
        for row in db.execute("SELECT * FROM jobs WHERE session_id=? AND state NOT IN ('completed','cancelled','failed')", (session_id,)).fetchall():
            self._terminate(db, row, "cancelled", {"code": code, "message": "client session ended"})

    def submit(self, principal, idem_key, request, releases):
        request = validate_request(request)
        from annoscript.ir import Program
        try:
            Program.from_json(encode(request["ir"]).decode())
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ServiceError("invalid_ir", "invalid AnnoScript IR") from exc
        fingerprint = request_sha256(request)
        with self._tx() as db:
            existing = db.execute("SELECT * FROM jobs WHERE principal=? AND idem_key=?", (principal, idem_key)).fetchone()
            if existing:
                if existing["request_hash"] != fingerprint:
                    raise ServiceError("idempotency_conflict", "idempotency key is bound to another request", 409)
                return self._status(db, self._job(db, existing["id"], principal))
            session_id = request.get("session_id")
            if session_id is not None and self._session(db, session_id, principal)["state"] != "active":
                raise ServiceError("session_closed", "cannot submit to a closed or expired session", 409)
            try:
                release_id, manifest = releases.resolve(request.get("release_id"))
            except KeyError:
                raise ServiceError("release_unavailable", "requested execution release is unavailable", 409) from None
            policy = manifest["policy"]
            if len(request["inputs"]) > policy["max_inputs"] or len(encode(request)) > policy["max_request_bytes"]:
                raise ServiceError("request_too_large", "job exceeds release resource limits", 413)
            db.execute("INSERT OR IGNORE INTO releases VALUES (?,?)", (release_id, encode(manifest, canonical=True)))
            job_id, now = uuid.uuid4().hex, self.clock()
            db.execute("""INSERT INTO jobs
                (id,principal,idem_key,request_hash,request,ir,digest,mode,release_id,session_id,state,indices,created,updated,metrics_available)
                VALUES (?,?,?,?,?,?,?,?,?,?,'queued',?,?,?,1)""",
                (job_id, principal, idem_key, fingerprint, encode(request), encode(request["ir"]), request["ir_sha256"],
                 request["mode"], release_id, session_id, encode([v["index"] for v in request["inputs"]]), now, now))
            db.execute("UPDATE jobs SET queued_monotonic=?,clock_epoch=? WHERE id=?",
                       (self.monotonic(), self.clock_epoch, job_id))
            db.executemany("INSERT INTO inputs(job_id,idx,state) VALUES (?,?,?)",
                           [(job_id, item["index"], encode(item["state"])) for item in request["inputs"]])
            return self._status(db, self._job(db, job_id))

    def lookup(self, principal, idem_key):
        with self._tx() as db:
            row = db.execute("SELECT id FROM jobs WHERE principal=? AND idem_key=?", (principal, idem_key)).fetchone()
            if row is None:
                raise ServiceError("not_found", "job not found", 404)
            return self._status(db, self._job(db, row["id"], principal))

    def _status(self, db, job):
        counts = {"success": 0, "error": 0}
        for row in db.execute("SELECT status,COUNT(*) AS n FROM results WHERE job_id=? GROUP BY status", (job["id"],)):
            counts[row["status"]] = row["n"]
        indices = decode(job["indices"])
        return {"job_id": job["id"], "ir_sha256": job["digest"], "release_id": job["release_id"],
                "mode": job["mode"], "input_indices": indices, "state": job["state"],
                "counts": {"total": len(indices), "completed": counts["success"] + counts["error"], **counts},
                "pause_generation": job["pause_generation"],
                "pause_reason": decode(job["pause_reason"]) if job["pause_reason"] is not None else None,
                "created_at": job["created"], "updated_at": job["updated"]}

    def status(self, principal, job_id):
        with self._tx() as db:
            return self._status(db, self._job(db, job_id, principal))

    def results(self, principal, job_id, cursor=None, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ServiceError("invalid_limit", "limit must be between 1 and 1000")
        try:
            after = cursor_seq(cursor, job_id)
        except ValueError as exc:
            raise ServiceError("invalid_cursor", str(exc)) from None
        with self._tx() as db:
            job = self._job(db, job_id, principal)
            if after > job["seq"]:
                raise ServiceError("invalid_cursor", "cursor is beyond the result log")
            rows = db.execute("SELECT * FROM results WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?", (job_id, after, limit)).fetchall()
            items = [{"seq": r["seq"], "index": r["idx"], "status": r["status"], "value": decode(r["value"]),
                      "error": decode(r["error"]), "attempts": r["attempts"]} for r in rows]
            last = items[-1]["seq"] if items else after
            more = last < job["seq"]
            return {"protocol_version": 1, "job": self._status(db, job), "results": items,
                    "next_cursor": cursor_for(job_id, last), "has_more": more,
                    "done": job["state"] in TERMINAL and not more}

    def release(self, release_id, releases):
        if release_id in releases.manifests:
            return {"release_id": release_id, "manifest": releases.manifests[release_id], "available": True}
        with self._tx() as db:
            row = db.execute("SELECT manifest FROM releases WHERE id=?", (release_id,)).fetchone()
            if row is None:
                raise ServiceError("not_found", "release not found", 404)
            return {"release_id": release_id, "manifest": decode(row["manifest"]), "available": False}

    def _policy(self, db, job):
        return decode(db.execute("SELECT manifest FROM releases WHERE id=?", (job["release_id"],)).fetchone()[0])["policy"]

    def _pause(self, db, job_id, reason):
        db.execute("""UPDATE jobs SET state='paused',pause_generation=pause_generation+1,pause_reason=?,updated=?
                      WHERE id=? AND state IN ('queued','running')""", (encode(reason), self.clock(), job_id))

    def _write_result(self, db, job_id, item, status, value, error):
        now = self.clock()
        db.execute("UPDATE jobs SET seq=seq+1,updated=? WHERE id=?", (now, job_id))
        seq = db.execute("SELECT seq FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
        db.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?)", (job_id, item["idx"], seq, status, encode(value), encode(error), item["total_attempts"]))
        db.execute("UPDATE inputs SET status='final',token=token+1,lease_until=NULL WHERE job_id=? AND idx=?", (job_id, item["idx"]))

    def _terminate(self, db, job, state, error):
        if job["state"] in TERMINAL:
            return
        now = self.clock()
        for item in db.execute("SELECT * FROM inputs WHERE job_id=? AND status!='final' ORDER BY idx", (job["id"],)).fetchall():
            self._write_result(db, job["id"], item, "error", None, error)
        db.execute("UPDATE attempts SET status='cancelled',finished=?,error=? WHERE job_id=? AND status='running'", (now, encode(error), job["id"]))
        db.execute("UPDATE jobs SET state=?,pause_reason=NULL,terminal_at=?,updated=? WHERE id=?", (state, now, now, job["id"]))

    def cancel(self, principal, job_id):
        with self._tx() as db:
            self._terminate(db, self._job(db, job_id, principal), "cancelled", {"code": "cancelled", "message": "job cancelled"})
            return self._status(db, self._job(db, job_id))

    def resume(self, principal, job_id, idem_key, generation, releases):
        with self._tx() as db:
            job = self._job(db, job_id, principal)
            old = db.execute("SELECT generation FROM resume_ops WHERE job_id=? AND idem_key=?", (job_id, idem_key)).fetchone()
            if old:
                if old[0] != generation:
                    raise ServiceError("idempotency_conflict", "resume key belongs to another pause generation", 409)
                return self._status(db, job)
            if job["state"] != "paused" or job["pause_generation"] != generation:
                raise ServiceError("pause_conflict", "job is not at the requested pause generation", 409)
            if job["release_id"] not in releases.manifests:
                raise ServiceError("release_unavailable", "original execution release is unavailable", 409)
            if db.execute("SELECT 1 FROM inputs WHERE job_id=? AND status='running'", (job_id,)).fetchone():
                raise ServiceError("attempts_active", "wait for existing attempts before resuming", 409)
            db.execute("INSERT INTO resume_ops VALUES (?,?,?)", (job_id, idem_key, generation))
            db.execute("UPDATE inputs SET round_attempts=0,next_attempt=0 WHERE job_id=? AND status!='final'", (job_id,))
            db.execute("UPDATE jobs SET state='queued',pause_reason=NULL,updated=? WHERE id=?", (self.clock(), job_id))
            return self._status(db, self._job(db, job_id))

    def _maintenance(self, db):
        now = self.clock()
        for session in db.execute("SELECT id FROM sessions WHERE state='active' AND expires<=?", (now,)).fetchall():
            db.execute("UPDATE sessions SET state='expired' WHERE id=?", (session["id"],))
            self._cancel_session_jobs(db, session["id"], "session_expired")
        for item in db.execute("SELECT * FROM inputs WHERE status='running' AND lease_until<=?", (now,)).fetchall():
            error = {"code": "lease_expired", "message": "worker attempt lease expired"}
            db.execute("UPDATE attempts SET status='abandoned',finished=?,error=? WHERE job_id=? AND idx=? AND number=?",
                       (now, encode(error), item["job_id"], item["idx"], item["total_attempts"]))
            db.execute("UPDATE inputs SET status='pending',token=token+1,lease_until=NULL,next_attempt=? WHERE job_id=? AND idx=?",
                       (now, item["job_id"], item["idx"]))
            job = self._job(db, item["job_id"])
            policy = self._policy(db, job)
            if (not policy.get("retry_unknown_outcomes", True)
                    or item["round_attempts"] >= policy["max_attempts"]):
                self._pause(db, job["id"], error)
        for row in db.execute("SELECT id FROM jobs WHERE expired=0 AND terminal_at<=?", (now - self.retention_seconds,)).fetchall():
            job_id = row["id"]
            for table in ("http_requests", "results", "attempts", "inputs", "resume_ops"):
                db.execute(f"DELETE FROM {table} WHERE job_id=?", (job_id,))
            db.execute("UPDATE jobs SET expired=1,request=NULL,ir=NULL,indices=NULL,pause_reason=NULL WHERE id=?", (job_id,))

    def claim(self, worker_id, releases):
        with self._tx() as db:
            for job in db.execute("SELECT * FROM jobs WHERE state IN ('queued','running') ORDER BY created,id").fetchall():
                if job["release_id"] not in releases.manifests:
                    self._pause(db, job["id"], {"code": "release_unavailable", "message": "original execution release is unavailable"})
                    continue
                item = db.execute("SELECT * FROM inputs WHERE job_id=? AND status='pending' AND next_attempt<=? ORDER BY idx LIMIT 1", (job["id"], self.clock())).fetchone()
                if item is None:
                    continue
                now, token, number = self.clock(), item["token"] + 1, item["total_attempts"] + 1
                db.execute("""UPDATE inputs SET status='running',round_attempts=round_attempts+1,total_attempts=?,
                              token=?,lease_until=? WHERE job_id=? AND idx=?""", (number, token, now + self.lease_seconds, job["id"], item["idx"]))
                db.execute("INSERT INTO attempts(job_id,idx,number,token,worker,status,started) VALUES (?,?,?,?,?,'running',?)",
                           (job["id"], item["idx"], number, token, worker_id, now))
                if number == 1 and self.clock_epoch is not None and self.clock_epoch == job["clock_epoch"]:
                    wait = self.monotonic() - job["queued_monotonic"]
                    db.execute("UPDATE attempts SET queue_wait_seconds=? WHERE job_id=? AND idx=? AND number=?",
                               (wait if wait >= 0 else None, job["id"], item["idx"], number))
                db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?", (now, job["id"]))
                return {"job_id": job["id"], "index": item["idx"], "state": decode(item["state"]),
                        "token": token, "attempt": number, "round_attempt": item["round_attempts"] + 1,
                        "ir": decode(job["ir"]), "mode": job["mode"], "release_id": job["release_id"]}
        return None

    def _owned(self, db, claim):
        job = db.execute("SELECT * FROM jobs WHERE id=?", (claim["job_id"],)).fetchone()
        item = db.execute("SELECT * FROM inputs WHERE job_id=? AND idx=?", (claim["job_id"], claim["index"])).fetchone()
        if (job is None or job["state"] in TERMINAL or item is None or item["status"] != "running"
                or item["token"] != claim["token"] or item["lease_until"] <= self.clock()):
            return None
        return job, item

    def renew(self, claim):
        with self._tx() as db:
            if self._owned(db, claim) is None:
                return False
            db.execute("UPDATE inputs SET lease_until=? WHERE job_id=? AND idx=?", (self.clock() + self.lease_seconds, claim["job_id"], claim["index"]))
            return True

    def can_dispatch(self, claim):
        """Pausing blocks new HTTP work, but preserves in-flight ownership."""
        with self._tx() as db:
            owned = self._owned(db, claim)
            return owned is not None and owned[0]["state"] in ("queued", "running")

    def pause_dispatch(self, claim, error):
        """Stop admission immediately, retaining leases for in-flight accounting."""
        with self._tx() as db:
            if self._owned(db, claim) is None:
                return False
            self._pause(db, claim["job_id"], error)
            return True

    def finish(self, claim, result, *, elapsed_seconds=None):
        if result.index != claim["index"]:
            raise ValueError("attempt result index mismatch")
        with self._tx() as db:
            owned = self._owned(db, claim)
            if owned is None:
                return False
            job, item = owned
            self._attempt_duration(db, claim, elapsed_seconds)
            error = None if result.error is None else {"code": result.error.code, "message": result.error.message}
            self._write_result(db, job["id"], item, result.status, result.value, error)
            db.execute("UPDATE attempts SET status='finished',finished=?,error=? WHERE job_id=? AND idx=? AND number=?",
                       (self.clock(), encode(error), job["id"], item["idx"], item["total_attempts"]))
            if not db.execute("SELECT 1 FROM inputs WHERE job_id=? AND status!='final'", (job["id"],)).fetchone():
                db.execute("UPDATE jobs SET state='completed',pause_reason=NULL,terminal_at=?,updated=? WHERE id=?", (self.clock(), self.clock(), job["id"]))
            return True

    def attempt_failed(self, claim, error, *, pause=False, delay=0.0, metadata=None, elapsed_seconds=None):
        with self._tx() as db:
            owned = self._owned(db, claim)
            if owned is None:
                return False
            job, item = owned
            self._attempt_duration(db, claim, elapsed_seconds)
            db.execute("UPDATE attempts SET status='failed',finished=?,error=?,metadata=? WHERE job_id=? AND idx=? AND number=?",
                       (self.clock(), encode(error), encode(metadata or {}), job["id"], item["idx"], item["total_attempts"]))
            db.execute("UPDATE inputs SET status='pending',lease_until=NULL,next_attempt=? WHERE job_id=? AND idx=?",
                       (self.clock() + delay, job["id"], item["idx"]))
            if pause or item["round_attempts"] >= self._policy(db, job)["max_attempts"]:
                self._pause(db, job["id"], error)
            return True

    def fail_job(self, claim, error, *, elapsed_seconds=None):
        with self._tx() as db:
            owned = self._owned(db, claim)
            if owned is None:
                return False
            self._attempt_duration(db, claim, elapsed_seconds)
            self._terminate(db, owned[0], "failed", error)
            return True
