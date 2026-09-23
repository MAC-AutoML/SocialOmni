import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from export_results import export


class ExportTests(unittest.TestCase):
    def test_incomplete_panel_is_explicit_and_private_endpoint_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, judges = root / "candidates", root / "judges"
            (candidates / "items").mkdir(parents=True)
            judges.mkdir()
            manifest = {
                "model": "test-model",
                "protocol": "test-protocol",
                "endpoint": "https://private.example/v1",
                "temperature": 0.3,
                "top_p": 1,
                "max_tokens": 8192,
                "enable_thinking": False,
                "prefix_encoding": {},
                "who_cutoff_rule": "interval-end",
                "source_sha256": {},
                "harness_sha256": {},
            }
            (candidates / "manifest.json").write_text(json.dumps(manifest))
            rows = (
                [
                    {"task": "who", "sample_id": str(i), "gold": "A", "prediction": "A"}
                    for i in range(2000)
                ]
                + [
                    {
                        "task": "when",
                        "sample_id": str(i),
                        "gold": "YES" if i < 128 else "NO",
                        "prediction": "NO",
                    }
                    for i in range(200)
                ]
                + [
                    {"task": "how", "sample_id": str(i), "text": "A continuation."}
                    for i in range(128)
                ]
            )
            for row in rows:
                (
                    candidates / "items" / f"{row['task']}-{row['sample_id']}.json"
                ).write_text(json.dumps(row))
            output = root / "public.json"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export(candidates, judges, output)
            self.assertFalse(output.exists())
            with contextlib.redirect_stdout(io.StringIO()):
                export(candidates, judges, output, allow_incomplete=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["metrics"]["who"], 100)
            self.assertEqual(result["metrics"]["when"], 36)
            self.assertIsNone(result["metrics"]["qgold"])
            self.assertEqual(result["metrics"]["cov_plus"], 0)
            self.assertNotIn("private.example", output.read_text())


if __name__ == "__main__":
    unittest.main()
