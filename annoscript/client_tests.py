"""Client-only suite: python -m annoscript.client_tests [-q|-v] (Python 3.10+).

An explicit module list avoids importing Runtime/Backend test modules during
discovery. Plan tests remain part of the full server-side regression suite.
"""

import unittest


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromNames([
        "annoscript.test_client", "annoscript.test_gateway_client", "annoscript.test_compiler", "annoscript.test_ir", "annoscript.test_passes", "annoscript.test_examples",
        "annoscript.test_release_client", "annoscript.test_metrics", "annoscript.test_formatter", "annoscript.test_remote", "annoscript.test_jobs", ])


if __name__ == "__main__":
    unittest.main()
