"""Client-only metric wire/usage contracts; no Runtime or server imports."""

from copy import deepcopy
import unittest

import httpx

from lev.jobs import LevJobClient
from lev.metrics import aggregate, normalize_usage, quantile, validate_metrics
from lev.remote import RemoteProtocolError


def fixture():
    job = {"job_id": "job", "ir_sha256": "a" * 64, "release_id": "b" * 64, "mode": "eager", "input_indices": [3],
           "state": "completed", "counts": {"total": 1, "completed": 1, "success": 1, "error": 0},
           "pause_generation": 0, "pause_reason": None, "created_at": 0, "updated_at": 1}
    attempt = {"metered": 1, "status": "finished", "duration_seconds": 1}
    request = {"backend": "generation", "questions": 1, "http_status": 200, "seconds": .5,
               "usage": normalize_usage({"prompt_tokens": 10, "completion_tokens": 2})}
    summary = aggregate([attempt], [request], history=True, waits=[.1])
    return {"protocol_version": 1, "metrics_version": 1, "job": job, "provisional": False,
            "summary": summary, "inputs": [{"index": 3, "metrics": deepcopy(summary)}]}


class MetricContracts(unittest.TestCase):
    def test_usage_aliases_details_missing_invalid_conflicts(self):
        for usage, value, reason in (({}, None, "missing"), ({"input_tokens": True}, None, "invalid"),
            ({"input_tokens": -1}, None, "invalid"), ({"input_tokens": 1.5}, None, "invalid"),
            ({"input_tokens": "1"}, None, "invalid"), ({"input_tokens": 1, "prompt_tokens": 2}, None, "conflicting"),
            ({"input_tokens": 2, "prompt_tokens": 2}, 2, None), ({"prompt_tokens": 0}, 0, None)):
            item = normalize_usage(usage)["input_tokens"]
            self.assertEqual((item["value"], item["reason"]), (value, reason))
        result = normalize_usage({"completion_tokens": 5, "total_tokens": 999, "completion_tokens_details": {"reasoning_tokens": 4}})
        self.assertEqual(result["output_tokens"]["value"], 5)

    def test_client_binding_and_invalid_responses(self):
        payload = fixture()
        with LevJobClient("http://offline", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
            self.assertEqual(client.metrics(payload["job"]), payload)
            changed = {**payload["job"], "release_id": "c" * 64}
            with self.assertRaises(RemoteProtocolError):
                client.metrics(changed)
        mutations = [lambda p: p.update(metrics_version=True), lambda p: p.update(provisional=True),
                     lambda p: p["job"].update(state=[]), lambda p: p.update(inputs=[None]),
                     lambda p: p["inputs"][0].update(index=4), lambda p: p["summary"]["backends"]["generation"].update(requests=2),
                     lambda p: p["summary"]["backends"]["generation"]["input_tokens"].update(total=0),
                     lambda p: p["summary"]["attempts"]["seconds"].update(p95=-1)]
        for mutation in mutations:
            bad = fixture(); mutation(bad)
            with self.assertRaises(RemoteProtocolError):
                validate_metrics(bad, "job")

    def test_unknown_history_never_becomes_zero(self):
        metric = aggregate([], [], history=False, waits=[None])
        self.assertIsNone(metric["backends"]["generation"]["input_tokens"]["total"])
        self.assertIsNone(metric["backends"]["generation"]["total_requests"])
        self.assertIsNone(metric["attempts"]["seconds"]["total"])
        self.assertEqual(quantile([1, 2, 3, 4], .95), 4)
        self.assertIsNone(quantile([], .5))
