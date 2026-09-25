import unittest

from judge_eval import metrics


class JudgeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.when = [
            {
                "sample_id": str(i),
                "gold": "YES" if i < 128 else "NO",
                "prediction": "YES",
            }
            for i in range(200)
        ]
        self.responses = [{"sample_id": str(i), "text": ""} for i in range(128)]
        self.responses[0]["text"] = "A continuation."

    def test_zero_scores_and_empty_responses(self):
        summary = metrics(
            self.when,
            self.responses,
            {"0": {"gpt-5.6-sol": 0, "gemini-3.8-flash": 0, "qwen3.8-omni": 0}},
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["qgold"], 0)
        self.assertEqual(summary["qens"], 0)
        self.assertEqual(summary["cov_plus"], 100 / 128)

    def test_incomplete_panel_never_averaged(self):
        summary = metrics(
            self.when,
            self.responses,
            {"0": {"gpt-5.6-sol": 100, "gemini-3.8-flash": 100}},
        )
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["qgold"])
        self.assertIsNone(summary["qens_joint"])

    def test_negative_entry_still_scores_forced_response(self):
        self.when[0]["prediction"] = "NO"
        summary = metrics(
            self.when,
            self.responses,
            {"0": {"gpt-5.6-sol": 75, "gemini-3.8-flash": 75, "qwen3.8-omni": 75}},
        )
        self.assertEqual(summary["qgold"], 75 / 128)
        self.assertEqual(summary["qens_joint"], 0)
        self.assertIsNone(summary["qens"])

    def test_transport_failure_cannot_be_complete(self):
        self.responses[0]["error"] = "transport failure"
        self.assertFalse(metrics(self.when, self.responses, {})["complete"])

    def test_substitute_panel_requires_explicit_selection(self):
        scores = {"0": {"gpt-5.6-sol": 100, "gemini-2.5-pro": 50, "qwen3-omni": 75}}
        self.assertFalse(metrics(self.when, self.responses, scores)["complete"])
        summary = metrics(self.when, self.responses, scores, "gpt56-sol")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["qgold"], 75 / 128)

    def test_original_panel_remains_explicit(self):
        scores = {"0": {"gpt-4o": 100, "gemini-2.5-pro": 50, "qwen3-omni": 75}}
        self.assertFalse(metrics(self.when, self.responses, scores)["complete"])
        self.assertTrue(
            metrics(self.when, self.responses, scores, "paper-v3")["complete"]
        )

    def test_original_scores_cannot_fill_substitute_panel(self):
        scores = {"0": {"gpt-4o": 100, "gemini-2.5-pro": 50, "qwen3-omni": 75}}
        self.assertFalse(
            metrics(self.when, self.responses, scores, "gpt56-sol")["complete"]
        )

    def test_minicpmo_run_label_keeps_standard_panel(self):
        scores = {
            "0": {
                "gpt-5.6-sol": 100,
                "gemini-3.8-flash": 50,
                "qwen3.8-omni": 75,
            }
        }
        summary = metrics(self.when, self.responses, scores, "minicpmo45-20260925")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["qgold"], 75 / 128)


if __name__ == "__main__":
    unittest.main()
