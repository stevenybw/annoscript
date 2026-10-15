"""In-memory session adapter behind LevClient.remote(); never imported locally."""

from collections import deque
import logging
import threading
import time
import uuid
import weakref

from lev._client_common import Result, ResultError, ResultStream, prepare_inputs
from lev.compiler import compile_program
from lev.jobs import (LevJobClient, RemoteCancellationError, RemoteJobPausedError,
                      RemoteProtocolError, RemoteSubmissionError, RemoteTransportError)


def bounded_cleanup(operation, timeout):
    """Bound caller latency even when DNS/transport shutdown ignores timeouts."""
    outcome, done = [], threading.Event()
    def run():
        try:
            operation()
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()
    threading.Thread(target=run, name="lev-job-cleanup", daemon=True).start()
    if not done.wait(timeout):
        raise RemoteCancellationError("remote cancellation confirmation timed out; session lease will expire")
    if outcome:
        raise RemoteCancellationError("remote cancellation could not be confirmed; session lease will expire") from outcome[0]


class _Session:
    def __init__(self, client):
        self.client = client
        self.id = None
        self.key = uuid.uuid4().hex
        self.closed = False
        self.stop = threading.Event()
        self.streams = weakref.WeakSet()
        self.failure = None
        self.thread = None

    def start(self):
        if self.closed:
            raise RuntimeError("client is closed")
        if self.failure is not None:
            raise self.failure
        if self.id is None:
            value = self.client._create_session(self.key)
            self.id = value["session_id"]
            self.thread = threading.Thread(target=self.heartbeat, args=(value["heartbeat_interval"],),
                                           name="lev-job-heartbeat", daemon=True)
            self.thread.start()

    def heartbeat(self, interval):
        # This target holds the session state, never the public LevClient shell.
        while not self.stop.wait(interval):
            try:
                self.client._heartbeat(self.id)
            except RemoteTransportError:
                continue
            except RemoteSubmissionError as exc:
                if exc.http_status in (408, 429) or (exc.http_status is not None and exc.http_status >= 500):
                    continue
                self.failure = exc
                return
            except Exception as exc:
                self.failure = exc
                return

    def close(self, timeout):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        for stream in list(self.streams):
            stream._closed = True
            stream.buffer.clear()
            stream.ready.clear()
        def cleanup():
            try:
                if self.id is not None:
                    self.client._close_session(self.id, timeout)
            finally:
                self.client.close()
        bounded_cleanup(cleanup, timeout)


def _finalize(session):
    try:
        session.close(1.0)
    except BaseException:
        # Destructors cannot report reliable cancellation. The durable session
        # expiry is the fallback, including when the process is killed outright.
        pass


class RemoteAdapter:
    def __init__(self, base_url, auth_token=None, *, mode="eager", optimize=True,
                 timeout=30.0, transport=None):
        if mode not in ("eager", "strict"):
            raise ValueError("mode must be eager or strict")
        if type(optimize) is not bool:
            raise TypeError("optimize must be a boolean")
        self.mode, self.optimize = mode, optimize
        self.session = _Session(LevJobClient(base_url, auth_token, timeout=timeout, transport=transport))

    @property
    def closed(self):
        return self.session.closed

    def submit(self, program, inputs, *, ordered=False):
        if self.closed:
            raise RuntimeError("client is closed")
        if type(ordered) is not bool:
            raise TypeError("ordered must be a boolean")
        valid, errors = prepare_inputs(program, inputs)
        ir = compile_program(program, optimize=self.optimize).to_dict()
        job = None
        if valid:
            self.session.start()
            try:
                job = self.session.client.submit_ir(ir, [{"index": i, "state": v} for i, v in valid],
                    idempotency_key=uuid.uuid4().hex, mode=self.mode, session_id=self.session.id)
            except BaseException:
                # Acceptance may have happened without its response. Closing
                # the ownership session also reaches such otherwise unknown jobs.
                try:
                    self.session.close(5.0)
                except BaseException:
                    logging.getLogger(__name__).warning("Unconfirmed remote cleanup after submission failure")
                raise
        stream = _JobStream(self.session, job, errors, ordered)
        self.session.streams.add(stream)
        return stream

    def close(self):
        self.session.close(5.0)


class _JobStream(ResultStream):
    def __init__(self, session, job, errors, ordered):
        super().__init__()
        self.session, self.job, self.ordered = session, job, ordered
        self.ready = deque(Result(i, "error", error=error) for i, error in sorted(errors.items()))
        self.buffer, self.index, self.seen = {}, 0, set()
        self.cursor = None
        self.done = job is None

    def _pull(self):
        while not self.ready:
            if self.done:
                raise StopIteration
            if self.session.closed:
                raise RemoteSubmissionError("client_closed", "client session is closed")
            if self.session.failure is not None:
                raise self.session.failure
            page = self.session.client.fetch_results(self.job, cursor=self.cursor)
            received = page["job"]
            for field in ("job_id", "ir_sha256", "release_id", "mode", "input_indices"):
                if received[field] != self.job[field]:
                    raise RemoteProtocolError("job binding changed while receiving results")
            pending = []
            for item in page["results"]:
                if item["index"] in self.seen:
                    raise RemoteProtocolError("result index was delivered more than once")
                pending.append(Result(item["index"], item["status"], item["value"],
                                      None if item["error"] is None else ResultError(**item["error"])))
            new_seen = self.seen | {r.index for r in pending}
            if page["done"] and new_seen != set(self.job["input_indices"]):
                raise RemoteProtocolError("terminal job is missing input results")
            # Commit a fully validated page to memory as a unit, then advance.
            self.ready.extend(pending)
            self.seen, self.cursor, self.done = new_seen, page["next_cursor"], page["done"]
            self.job = received
            if not self.ready:
                if received["state"] == "paused":
                    raise RemoteJobPausedError(received)
                if not self.done:
                    self.session.stop.wait(0.2)
        return self.ready.popleft()

    def _next_result(self):
        if not self.ordered:
            return self._pull()
        while self.index not in self.buffer:
            item = self._pull()
            self.buffer[item.index] = item
        item = self.buffer.pop(self.index)
        self.index += 1
        return item

    def _close(self):
        deadline = time.monotonic() + 5.0
        try:
            if self.job is not None and not self.done and not self.session.closed:
                try:
                    bounded_cleanup(lambda: self.session.client.cancel(self.job, attempts=1, budget=5.0), 5.0)
                except RemoteCancellationError:
                    # Stop owning work if even cancellation cannot be delivered.
                    # Do not keep heartbeating an orphaned job indefinitely.
                    try:
                        self.session.close(max(0.0, deadline - time.monotonic()))
                    except RemoteCancellationError:
                        pass
                    raise
        finally:
            self.ready.clear()
            self.buffer.clear()
            self.session.streams.discard(self)
