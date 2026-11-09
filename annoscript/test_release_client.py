import hashlib
import unittest

import httpx

from annoscript.job_protocol import encode
from annoscript.jobs import AnnoJobClient, RemoteProtocolError


class ReleaseTests(unittest.TestCase):
    def test_digest_available_and_unavailable(self):
        manifest = {"format": "lev-release", "version": 1, "policy": {"max_inputs": 500}}
        rid = hashlib.sha256(encode(manifest, canonical=True)).hexdigest()
        for available in (False, True):
            payload = {"protocol_version": 1, "release_id": rid, "manifest": manifest, "available": available}
            with AnnoJobClient("http://offline", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as client:
                self.assertEqual(client.release(rid), payload)
                payload["manifest"] = {"changed": True}
                with self.assertRaises(RemoteProtocolError):
                    client.release(rid)
