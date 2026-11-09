"""Detached inspection of the exact HTTP body; never includes transport headers."""

import hashlib

from .base import encode_payload


def request_preview(payload, *, stage, adapter, url, evidence=None):
    body = encode_payload(payload)
    return {"format": "lev-request-preview", "version": 1, "stage": stage,
            "request_adapter": adapter, "method": "POST", "url": url,
            "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
            "payload": payload, "evidence": evidence or {}}
