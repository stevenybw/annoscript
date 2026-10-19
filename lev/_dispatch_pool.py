"""Event-loop-local, FIFO model slots shared by isolated input runtimes.

Waiters contain owner identities and wake callbacks, never model calls or payloads.
Batch selection therefore happens after admission, using the current ready set.
"""
from collections import OrderedDict


class RequestSlots:
    def __init__(self, *, generation, decision):
        if any(type(v) is not int or v < 1 for v in (generation, decision)):
            raise ValueError("shared model limits must be positive integers")
        self.limits = {"llm": generation, "decision": decision}
        self.active = dict.fromkeys(self.limits, 0)
        self.peak = dict(self.active)
        self.waiters = {k: OrderedDict() for k in self.limits}

    def acquire(self, kind, owner, wake):
        waiting = self.waiters[kind]
        waiting.setdefault(owner, wake)
        if self.active[kind] >= self.limits[kind] or next(iter(waiting)) != owner:
            return None
        waiting.pop(owner)
        self.active[kind] += 1
        self.peak[kind] = max(self.peak[kind], self.active[kind])
        self._wake(kind)
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                self.active[kind] -= 1
                self._wake(kind)
        return release

    def _wake(self, kind):
        for wake in list(self.waiters[kind].values())[:self.limits[kind] - self.active[kind]]:
            wake()

    def cancel(self, owner):
        for kind in self.limits:
            if self.waiters[kind].pop(owner, None) is not None:
                self._wake(kind)

    def waiting(self, owner):
        return any(owner in q for q in self.waiters.values())
