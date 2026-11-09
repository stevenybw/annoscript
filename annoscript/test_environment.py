"""Configuration compatibility across real file/process boundaries; no network."""

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annoscript._env import env_value
from annoscript.backends.config import backends_from_env


class EnvironmentTests(unittest.TestCase):
    def configured(self, file_values, environment, *, generation_profile=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration"
            path.write_text("\n".join(f"{k}={v}" for k, v in file_values.items()))
            with patch.dict(os.environ, environment, clear=True):
                decision, generation = backends_from_env(env_file=path, generation_profile=generation_profile)
                try:
                    return decision.model, generation.model, generation.profile
                finally:
                    async def close():
                        await decision.aclose()
                        await generation.aclose()
                    asyncio.run(close())

    def baseline(self, prefix):
        return {f"{prefix}_{stage}_{key}": value
                for stage in ("DECISION", "GENERATE")
                for key, value in (("BASE_URL", "https://example.invalid/v1"),
                                   ("AUTH_TOKEN", "fixture"), ("MODEL", "file-model"))}

    def test_legacy_file_remains_usable(self):
        self.assertEqual(self.configured(self.baseline("LEV"), {}),
                         ("file-model", "file-model", "legacy"))

    def test_legacy_process_overrides_canonical_file(self):
        self.assertEqual(self.configured(self.baseline("ANNO"), {"LEV_GENERATE_MODEL": "process"})[1],
                         "process")

    def test_canonical_process_overrides_legacy_file(self):
        self.assertEqual(self.configured(self.baseline("LEV"), {"ANNO_GENERATE_MODEL": "process"})[1],
                         "process")

    def test_canonical_wins_within_each_layer(self):
        file = {**self.baseline("LEV"), "ANNO_GENERATE_MODEL": "canonical-file"}
        self.assertEqual(self.configured(file, {})[1], "canonical-file")
        self.assertEqual(self.configured(file, {"LEV_GENERATE_MODEL": "old", "ANNO_GENERATE_MODEL": "new"})[1], "new")

    def test_explicit_arguments_override_both_layers(self):
        file = {**self.baseline("LEV"), "ANNO_GENERATE_PROFILE": "invalid"}
        self.assertEqual(self.configured(file, {"LEV_GENERATE_PROFILE": "also-invalid"},
                                         generation_profile="qwen_vllm")[2], "qwen_vllm")

    def test_empty_canonical_value_does_not_restore_legacy_secret(self):
        with self.assertRaisesRegex(ValueError, "ANNO_GENERATE_AUTH_TOKEN"):
            self.configured(self.baseline("LEV"), {"ANNO_GENERATE_AUTH_TOKEN": ""})
        self.assertEqual(env_value("ANNO_SERVICE_TOKEN", environ={"LEV_SERVICE_TOKEN": "old"}), "old")
        self.assertEqual(env_value("ANNO_SERVICE_TOKEN", environ={"LEV_SERVICE_TOKEN": "old", "ANNO_SERVICE_TOKEN": "new"}), "new")


if __name__ == "__main__":
    unittest.main()
