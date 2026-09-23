import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from client import digest
from export_results import export


class ExportTests(unittest.TestCase):
    def make_scored_run(self, root):
        candidates, judges = root / "candidates", root / "judges"
        (candidates / "items").mkdir(parents=True)
        judges.mkdir()
        manifest = {
            "model": "test-model",
            "protocol": "test-protocol",
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
                    "prediction": "YES" if i < 128 else "NO",
                }
                for i in range(200)
            ]
            + [
                {"task": "how", "sample_id": str(i), "text": "A continuation."}
                for i in range(128)
            ]
        )
        for row in rows:
            path = candidates / "items" / f"{row['task']}-{row['sample_id']}.json"
            path.write_text(json.dumps(row))
        how = [
            json.loads(path.read_text())
            for path in sorted((candidates / "items").glob("how-*.json"))
        ]
        (judges / "manifest.json").write_text(
            json.dumps(
                {
                    "panel": "gpt56-sol",
                    "candidate_manifest_sha256": digest(manifest),
                    "candidate_rows_sha256": digest(how),
                }
            )
        )
        (judges / "scores.json").write_text(
            json.dumps(
                {
                    row["sample_id"]: {
                        "gpt-5.6-sol": 100,
                        "gemini-2.5-pro": 75,
                        "qwen3-omni": 50,
                    }
                    for row in how
                }
            )
        )
        return candidates, judges

    def test_sol_panel_export_and_failed_who_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, judges = self.make_scored_run(root)
            output = root / "public.json"
            with contextlib.redirect_stdout(io.StringIO()):
                export(candidates, judges, output)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["judge_panel"], "gpt56-sol")
            self.assertEqual(
                result["judge_substitution"],
                {
                    "replaced": "gpt-4o",
                    "replacement": "gpt-5.6-sol",
                },
            )
            self.assertEqual(result["judge_score_count"], 384)
            self.assertEqual(result["metrics"]["qgold"], 75)
            self.assertEqual(result["metrics"]["qens_joint"], 75)

            failed = candidates / "items" / "who-0.json"
            row = json.loads(failed.read_text())
            row.update(prediction=None, error="HTTP 400")
            failed.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export(candidates, judges, output)
            with contextlib.redirect_stdout(io.StringIO()):
                export(candidates, judges, output, allow_incomplete=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "scored_with_request_failures")
            self.assertEqual(result["metrics"]["who"], 99.95)
            self.assertEqual(result["metrics"]["qgold"], 75)
            self.assertEqual(result["judge_score_count"], 384)

    def test_changed_candidate_inputs_reject_existing_scores(self):
        for changed in ("manifest.json", "items/how-0.json"):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                candidates, judges = self.make_scored_run(root)
                path = candidates / changed
                value = json.loads(path.read_text())
                value["model" if changed == "manifest.json" else "text"] = "Changed"
                path.write_text(json.dumps(value))
                output = root / "public.json"
                with self.assertRaisesRegex(ValueError, "do not match"):
                    export(candidates, judges, output, allow_incomplete=True)
                self.assertFalse(output.exists())

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
