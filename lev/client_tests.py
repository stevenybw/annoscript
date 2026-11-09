"""Client-only suite: python -m lev.client_tests [-q|-v] (Python 3.10+).

An explicit module list avoids importing Runtime/Backend test modules during
discovery. Plan tests remain part of the full server-side regression suite.
"""

import unittest


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromNames([
        "lev.test_client", "lev.test_compiler", "lev.test_ir", "lev.test_passes", "lev.test_examples",
        "lev.test_release_client", "lev.test_metrics", "lev.test_formatter", "lev.test_remote", "lev.test_jobs", ])


if __name__ == "__main__":
    unittest.main()
