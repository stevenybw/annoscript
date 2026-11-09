"""Version 1 metric values and validation (stdlib only, Python 3.10+).

Observed subtotals are never substituted for unknown totals. Token details such
as reasoning/cached tokens are subsets, not additional billable tokens.
"""

from collections import Counter
import math

from annoscript.job_protocol import TERMINAL, integer, validate_status
from annoscript.remote import RemoteProtocolError


def normalize_usage(usage):
    result = {}
    for field, aliases in (("input_tokens", ("input_tokens", "prompt_tokens")),
                           ("output_tokens", ("output_tokens", "completion_tokens"))):
        present = [key for key in aliases if type(usage) is dict and key in usage]
        values = [usage[key] for key in present]
        reason = ("missing" if not present else "invalid" if not all(integer(v) for v in values)
                  else "conflicting" if len(set(values)) != 1 else None)
        result[field] = {"value": values[0] if reason is None else None,
                         "sources": present, "reason": reason}
    return result


def quantile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def measure(values, *, history=True, duration=False):
    known = [value for value in values if value is not None]
    result = {"observed": sum(known), "total": sum(known) if history and len(known) == len(values) else None,
              "known": len(known), "unknown": len(values) - len(known)}
    if duration:
        result.update(p50=quantile(known, .5), p95=quantile(known, .95))
    return result


def aggregate(attempts, requests, *, history, waits):
    """Aggregate sanitized persisted records; never distribute batch usage by question."""
    history = history and all(a["metered"] for a in attempts)
    result = {
        "history_complete": bool(history),
        "attempts": {"count": len(attempts), "statuses": dict(sorted(Counter(a["status"] for a in attempts).items())),
                     "seconds": measure([a["duration_seconds"] for a in attempts], history=history, duration=True)},
        "queue_wait_seconds": measure(waits, history=history, duration=True),
        "backends": {},
    }
    for backend in ("decision", "generation"):
        rows = [r for r in requests if r["backend"] == backend]
        unknown = sum(r["http_status"] is None for r in rows)
        item = {"requests": len(rows), "confirmed_requests": len(rows) - unknown,
                "unknown_requests": unknown, "total_requests": len(rows) if history and not unknown else None,
                "questions": sum(r["questions"] for r in rows),
                "confirmed_questions": sum(r["questions"] for r in rows if r["http_status"] is not None),
                "total_questions": sum(r["questions"] for r in rows) if history and not unknown else None,
                "http_seconds": measure([r["seconds"] for r in rows], history=history, duration=True)}
        for field in ("input_tokens", "output_tokens"):
            tokens = [r["usage"][field] for r in rows]
            item[field] = {**measure([t["value"] for t in tokens], history=history),
                           "sources": sorted({s for t in tokens for s in t["sources"]}),
                           "reasons": dict(sorted(Counter(t["reason"] for t in tokens if t["reason"]).items()))}
        result["backends"][backend] = item
    return result


def validate_metrics(value, job_id):
    """Reject all malformed JSON shapes through the public protocol error type."""
    try:
        return _validate_metrics(value, job_id)
    except RemoteProtocolError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        raise RemoteProtocolError("invalid job metrics") from None


def _validate_metrics(value, job_id):
    def require(ok):
        if not ok:
            raise RemoteProtocolError("invalid job metrics")

    def number(v):
        return type(v) in (int, float) and 0 <= v < float("inf")

    def metric(m, *, duration=False, tokens=False):
        keys = {"observed", "total", "known", "unknown"}
        if duration:
            keys |= {"p50", "p95"}
        if tokens:
            keys |= {"sources", "reasons"}
        require(type(m) is dict and set(m) == keys)
        require(integer(m["known"]) and integer(m["unknown"]) and number(m["observed"]))
        require(m["total"] is None or (number(m["total"]) and m["total"] == m["observed"] and not m["unknown"]))
        require(m["known"] != 0 or m["observed"] == 0)
        if tokens:
            require(integer(m["observed"]) and (m["total"] is None or integer(m["total"])))
            require(type(m["sources"]) is list and all(s in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens") for s in m["sources"]))
            require(len(set(m["sources"])) == len(m["sources"]))
            require(type(m["reasons"]) is dict and all(k in ("missing", "invalid", "conflicting") and integer(v, 1) for k, v in m["reasons"].items()))
            require(sum(m["reasons"].values()) == m["unknown"])
        if duration:
            require(all((m[k] is None if not m["known"] else number(m[k])) for k in ("p50", "p95")))
            require(not m["known"] or m["p50"] <= m["p95"] <= m["observed"] + 1e-9)

    def summary(s):
        require(type(s) is dict and set(s) == {"history_complete", "attempts", "queue_wait_seconds", "backends"})
        require(type(s["history_complete"]) is bool)
        a = s["attempts"]
        require(type(a) is dict and set(a) == {"count", "statuses", "seconds"} and integer(a["count"]))
        require(type(a["statuses"]) is dict and all(k in ("running", "finished", "failed", "cancelled", "abandoned") and integer(v, 1) for k, v in a["statuses"].items()))
        require(sum(a["statuses"].values()) == a["count"])
        metric(a["seconds"], duration=True)
        require(a["seconds"]["known"] + a["seconds"]["unknown"] == a["count"])
        metric(s["queue_wait_seconds"], duration=True)
        require(s["history_complete"] or (a["seconds"]["total"] is None and s["queue_wait_seconds"]["total"] is None))
        require(type(s["backends"]) is dict and set(s["backends"]) == {"decision", "generation"})
        for b in s["backends"].values():
            require(type(b) is dict and set(b) == {"requests", "confirmed_requests", "unknown_requests", "total_requests", "questions", "confirmed_questions", "total_questions", "http_seconds", "input_tokens", "output_tokens"})
            require(all(integer(b[k]) for k in ("requests", "confirmed_requests", "unknown_requests", "questions", "confirmed_questions")))
            require(b["confirmed_requests"] + b["unknown_requests"] == b["requests"] and b["confirmed_questions"] <= b["questions"])
            complete = s["history_complete"] and not b["unknown_requests"]
            require(all(b[k] is None or integer(b[k]) for k in ("total_requests", "total_questions")))
            require(b["total_requests"] == (b["requests"] if complete else None))
            require(b["total_questions"] == (b["questions"] if complete else None))
            for key in ("http_seconds", "input_tokens", "output_tokens"):
                metric(b[key], duration=key == "http_seconds", tokens=key != "http_seconds")
                require(b[key]["known"] + b[key]["unknown"] == b["requests"])
                require(s["history_complete"] or b[key]["total"] is None)
        return s

    require(type(value) is dict and set(value) == {"protocol_version", "metrics_version", "job", "provisional", "summary", "inputs"})
    require(type(value["protocol_version"]) is int and value["protocol_version"] == 1)
    require(type(value["metrics_version"]) is int and value["metrics_version"] == 1)
    job = validate_status(value["job"])
    require(job["job_id"] == job_id and type(value["provisional"]) is bool and value["provisional"] == (job["state"] not in TERMINAL))
    summary(value["summary"])
    require(type(value["inputs"]) is list and len(value["inputs"]) == len(job["input_indices"]))
    indices = []
    for item in value["inputs"]:
        require(type(item) is dict and set(item) == {"index", "metrics"} and integer(item["index"]))
        indices.append(item["index"])
        summary(item["metrics"])
    require(indices == job["input_indices"])
    require(value["summary"]["attempts"]["count"] == sum(i["metrics"]["attempts"]["count"] for i in value["inputs"]))
    for backend in ("decision", "generation"):
        for k in ("requests", "questions", "unknown_requests", "confirmed_requests", "confirmed_questions"):
            require(value["summary"]["backends"][backend][k] == sum(i["metrics"]["backends"][backend][k] for i in value["inputs"]))
        for k in ("input_tokens", "output_tokens", "http_seconds"):
            observed = sum(i["metrics"]["backends"][backend][k]["observed"] for i in value["inputs"])
            actual = value["summary"]["backends"][backend][k]["observed"]
            require(math.isclose(actual, observed, rel_tol=1e-12, abs_tol=1e-9) if k == "http_seconds" else actual == observed)
    return value
