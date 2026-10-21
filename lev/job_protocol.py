"""Python 3.10 job wire contracts. No transport or execution dependencies."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from email.utils import parsedate_to_datetime

from lev._data import snapshot
from lev.remote import RemoteProtocolError, _json, ir_sha256

PROTOCOL_VERSION = 1
TERMINAL = frozenset(("completed", "cancelled", "failed"))
STATES = TERMINAL | {"queued", "running", "paused"}


def encode(value, *, canonical=False):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=canonical,
                      separators=(",", ":")).encode("utf-8")


def decode(value):
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except UnicodeError:
            raise RemoteProtocolError("JSON must be UTF-8") from None
    return _json(value)


def nonempty(value):
    return type(value) is str and bool(value.strip())


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_request(value):
    value = snapshot(value)
    required = {"protocol_version", "ir", "ir_sha256", "mode", "inputs"}
    optional = {"release_id", "application_revision", "session_id"}
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("invalid job request fields")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != 1:
        raise ValueError("unsupported job protocol version")
    if type(value["ir"]) is not dict or ir_sha256(value["ir"]) != value["ir_sha256"]:
        raise ValueError("IR digest mismatch")
    if value["mode"] not in ("eager", "strict", "speculative"):
        raise ValueError("invalid execution mode")
    if type(value["inputs"]) is not list or not value["inputs"]:
        raise ValueError("a job needs at least one input")
    indices = set()
    for item in value["inputs"]:
        if (type(item) is not dict or set(item) != {"index", "state"}
                or not integer(item["index"]) or item["index"] in indices):
            raise ValueError("inputs require unique nonnegative indices and states")
        snapshot(item["state"], state=True)
        indices.add(item["index"])
    for name in optional & value.keys():
        if not nonempty(value[name]):
            raise ValueError(f"{name} must be a nonempty string")
    return value


def request_sha256(request):
    # Normalize protocol field order and IR encoding, but NEVER sort state keys:
    # an IR program can iterate over the keys of an input object.
    value = {key: request[key] for key in ("protocol_version", "ir_sha256", "mode")}
    value["inputs"] = [{"index": item["index"], "state": item["state"]}
                       for item in request["inputs"]]
    for key in ("release_id", "application_revision", "session_id"):
        if key in request:
            value[key] = request[key]
    return hashlib.sha256(encode(value)).hexdigest()


def cursor_for(job_id, seq):
    return base64.urlsafe_b64encode(encode([1, job_id, seq])).decode("ascii").rstrip("=")


def cursor_seq(cursor, job_id):
    if cursor is None:
        return 0
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        value = decode(raw)
        if (type(value) is not list or len(value) != 3 or type(value[0]) is not int
                or value[0] != 1 or value[1] != job_id or not integer(value[2])
                or cursor_for(job_id, value[2]) != cursor):
            raise ValueError()
        return value[2]
    except (ValueError, TypeError, UnicodeError, RemoteProtocolError):
        raise ValueError("invalid cursor for this job") from None


def retry_after_seconds(value, now):
    if value is None:
        return 0.0
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0.0, seconds)
    except (ValueError, TypeError):
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - now)
    except (ValueError, TypeError, OverflowError):
        return 0.0


def validate_status(job):
    from lev.remote import _error
    fields = {"job_id", "ir_sha256", "release_id", "mode", "input_indices", "state", "counts",
              "pause_generation", "pause_reason", "created_at", "updated_at"}
    if type(job) is not dict or set(job) != fields:
        raise RemoteProtocolError("invalid job status fields")
    if (not all(nonempty(job[k]) for k in ("job_id", "ir_sha256", "release_id"))
            or len(job["ir_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in job["ir_sha256"])
            or job["mode"] not in ("strict", "eager", "speculative")
            or job["state"] not in STATES or not integer(job["pause_generation"])):
        raise RemoteProtocolError("invalid job status")
    indices, counts = job["input_indices"], job["counts"]
    if (type(indices) is not list or not indices or any(not integer(i) for i in indices)
            or len(set(indices)) != len(indices) or type(counts) is not dict
            or set(counts) != {"total", "completed", "success", "error"}
            or any(not integer(n) for n in counts.values())
            or counts["total"] != len(indices)
            or counts["completed"] != counts["success"] + counts["error"]
            or counts["completed"] > counts["total"]
            or ((job["state"] in TERMINAL) != (counts["completed"] == counts["total"]))):
        raise RemoteProtocolError("invalid job indices or counts")
    if job["pause_reason"] is not None:
        _error(job["pause_reason"])
    if (job["state"] == "paused") != (job["pause_reason"] is not None):
        raise RemoteProtocolError("inconsistent paused job")
    if any(type(job[k]) not in (int, float) or not math.isfinite(job[k])
           for k in ("created_at", "updated_at")):
        raise RemoteProtocolError("invalid job timestamps")
    return job


def validate_page(page, job_id, cursor=None):
    from lev._client_common import Result
    from lev.remote import _error
    if (type(page) is not dict
            or set(page) != {"protocol_version", "job", "results", "next_cursor", "has_more", "done"}
            or type(page["protocol_version"]) is not int or page["protocol_version"] != 1
            or type(page["results"]) is not list
            or type(page["has_more"]) is not bool or type(page["done"]) is not bool):
        raise RemoteProtocolError("invalid result page")
    job = validate_status(page["job"])
    if job["job_id"] != job_id:
        raise RemoteProtocolError("page belongs to another job")
    try:
        last = cursor_seq(cursor, job_id)
        next_seq = cursor_seq(page["next_cursor"], job_id)
    except ValueError as exc:
        raise RemoteProtocolError(str(exc)) from None
    seen, expected = set(), set(job["input_indices"])
    for item in page["results"]:
        if (type(item) is not dict or set(item) != {"seq", "index", "status", "value", "error", "attempts"}
                or not integer(item["seq"], 1) or item["seq"] != last + 1
                or not integer(item["index"]) or item["index"] not in expected or item["index"] in seen
                or not integer(item["attempts"])):
            raise RemoteProtocolError("invalid, missing or duplicate page result")
        try:
            Result(item["index"], item["status"], item["value"],
                   None if item["error"] is None else _error(item["error"]))
        except ValueError as exc:
            raise RemoteProtocolError(str(exc)) from None
        last = item["seq"]
        seen.add(item["index"])
    if (next_seq != last or last > job["counts"]["completed"]
            or page["has_more"] != (last < job["counts"]["completed"])
            or page["done"] != (job["state"] in TERMINAL and not page["has_more"])):
        raise RemoteProtocolError("inconsistent result cursor or completion")
    return page
