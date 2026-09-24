import unittest

from rescore import record_key, summarize, validate

PANEL = {"one", "two", "three"}


def fixture(full_gold=False):
    ids = [f"{i:04d}" for i in range(200)]
    expected = ids[:128] if full_gold else ids[:2]
    return {
        "schema_version": 1,
        "source": "test",
        "models": [
            {
                "model": "candidate",
                "full_gold": full_gold,
                "expected_response_ids": expected,
                "when": [
                    {
                        "item_id": item,
                        "gold": "YES" if i < 128 else "NO",
                        "prediction": "YES" if i < 2 else "NO",
                    }
                    for i, item in enumerate(ids)
                ],
            }
        ],
        "records": [
            {
                "model": "candidate",
                "item_id": item,
                "context": "context",
                "target_question": "continue",
                "reference": "reference",
                "candidate": "answer",
            }
            for item in expected
        ],
    }


def score_all(data, value):
    return {
        record_key(row): {judge: {"score": value} for judge in PANEL}
        for row in data["records"]
    }


class RescoreTests(unittest.TestCase):
    def test_legacy_does_not_invent_gold_quality(self):
        data = fixture()
        result = summarize(data, score_all(data, 75), PANEL)[0]
        self.assertTrue(result["complete"])
        self.assertIsNone(result["qgold"])
        self.assertEqual(result["qens"], 75)
        self.assertEqual(result["qens_joint"], 150 / 128)
        self.assertEqual(result["cov_plus"], 200 / 128)

    def test_complete_zero_is_valid(self):
        data = fixture(True)
        result = summarize(data, score_all(data, 0), PANEL)[0]
        self.assertTrue(result["complete"])
        self.assertEqual(result["qgold"], 0)
        self.assertEqual(result["qens"], 0)

    def test_missing_panel_does_not_average_remaining_judges(self):
        data = fixture(True)
        scores = score_all(data, 100)
        del scores[record_key(data["records"][0])]["one"]
        result = summarize(data, scores, PANEL)[0]
        self.assertFalse(result["complete"])
        for name in ("qgold", "qens", "qens_joint"):
            self.assertIsNone(result[name])
        self.assertEqual(result["covered_positive"], 2)

    def test_empty_success_is_zero_not_missing_judging(self):
        data = fixture(True)
        for row in data["records"]:
            row["candidate"] = ""
        result = summarize(data, {}, PANEL)[0]
        self.assertTrue(result["complete"])
        self.assertEqual(result["qgold"], 0)
        self.assertEqual(result["qens_joint"], 0)
        self.assertIsNone(result["qens"])

    def test_incomplete_or_wrong_candidate_set_rejected(self):
        data = fixture(True)
        data["records"].pop()
        with self.assertRaises(ValueError):
            validate(data)
        data = fixture()
        data["records"][0]["item_id"] = "0002"
        with self.assertRaises(ValueError):
            validate(data)
        data = fixture()
        data["models"][0]["when"][0]["prediction"] = "NO"
        with self.assertRaises(ValueError):
            validate(data)


if __name__ == "__main__":
    unittest.main()
