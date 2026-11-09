"""Worker-local bounded plan cache. Never contains input or inference state."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

from annoscript.job_protocol import encode

MAX_ENTRIES = 32
MAX_BYTES = 64 * 1024 * 1024


class PlanCache:
    def __init__(self, *, max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries, self.pending = OrderedDict(), {}
        self.bytes = 0
        self.stats = dict(hits=0, misses=0, waits=0, compiles=0, failures=0, evictions=0)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anno-plan")
        self.closed = False

    async def get(self, release_id, ir, compile_plan):
        if self.closed:
            raise RuntimeError("plan cache is closed")
        started = time.monotonic()
        encoded = encode(ir, canonical=True)
        key = (release_id, hashlib.sha256(encoded).hexdigest())
        status = "hit"
        if key in self.entries:
            self.stats["hits"] += 1
            entry = self.entries[key]
            self.entries.move_to_end(key)
        else:
            status = "wait" if key in self.pending else "miss"
            self.stats["waits" if status == "wait" else "misses"] += 1
            if status == "miss":
                self.pending[key] = asyncio.create_task(self._compile(key, encoded, compile_plan))
                # Retrieve exceptions even if every input waiting for this plan was cancelled.
                self.pending[key].add_done_callback(lambda t: None if t.cancelled() else t.exception())
            entry = await asyncio.shield(self.pending[key])
        plan, size, compile_seconds = entry
        return plan, {"event": "plan_cache", "status": status, "ir_sha256": key[1],
                      "seconds": time.monotonic() - started,
                      "compile_seconds": compile_seconds if status != "hit" else 0.0,
                      "plan_bytes": size, "cached_bytes": self.bytes,
                      "cached_entries": len(self.entries), "counters": dict(self.stats)}

    async def _compile(self, key, encoded, compile_plan):
        def build():
            started = time.monotonic()
            plan = compile_plan(encoded)
            size = len(plan.to_json(indent=None).encode("utf-8"))
            return plan, size, time.monotonic() - started
        try:
            entry = await asyncio.get_running_loop().run_in_executor(self.executor, build)
            self.stats["compiles"] += 1
            size = entry[1]
            if self.max_entries > 0 and size <= self.max_bytes:
                while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old[1]
                    self.stats["evictions"] += 1
                self.entries[key] = entry
                self.bytes += size
            return entry
        except Exception:
            self.stats["failures"] += 1
            raise
        finally:
            self.pending.pop(key, None)

    async def aclose(self):
        self.closed = True
        await asyncio.gather(*self.pending.values(), return_exceptions=True)
        self.executor.shutdown(wait=True)
        self.entries.clear()
        self.bytes = 0
