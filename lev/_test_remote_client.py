"""Python 3.10 side of the offline cross-process IR acceptance test."""

import importlib.abc
import json
from pathlib import Path
import sys


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("lev.runtime", "lev.plan", "lev.plan_lowering") or fullname.startswith("lev.backends"):
            raise AssertionError("client loaded execution code: " + fullname)


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("test client requires Python 3.10")
    sys.meta_path.insert(0, ImportGuard())
    from lev_examples.request import annotate
    from lev.remote import RemoteLevClient

    fixture = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    for mode in ("strict", "eager", "speculative"):
        for optimize in (False, True):
            for ordered in (False, True):
                with RemoteLevClient(sys.argv[1], mode=mode, optimize=optimize, timeout=30) as client:
                    results = list(client.submit(annotate, fixture["inputs"], ordered=ordered))
                if ordered and [r.index for r in results] != list(range(len(fixture["inputs"]))):
                    raise AssertionError("ordered results lost input indices")
                indexed = {r.index: r for r in results}
                if len(results) != len(indexed) or len(indexed) != len(fixture["expected"]):
                    raise AssertionError("missing or duplicate results")
                for index, expected in enumerate(fixture["expected"]):
                    result = indexed[index]
                    if result.status != expected["status"]:
                        raise AssertionError((mode, optimize, index, result, expected))
                    if result.status == "success":
                        if result.value != expected["value"]:
                            raise AssertionError((index, result.value, expected["value"]))
                    elif result.error.code != expected["code"] or result.value is not None:
                        raise AssertionError((index, result, expected))
                print(f"mode={mode}, optimize={optimize}, ordered={ordered}: {len(results)} indexed outcomes OK", flush=True)


if __name__ == "__main__":
    main()
