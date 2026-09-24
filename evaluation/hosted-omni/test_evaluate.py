import argparse
import json
import unittest
from pathlib import Path

from evaluate import (
    apply_model_defaults,
    gold_when,
    media_input,
    participant,
    seconds,
    summarize,
    who_cutoff,
)


class ProtocolTests(unittest.TestCase):
    def test_model_request_defaults_and_overrides(self):
        for model in (
            "dashscope/qwen3.8-omni-flash",
            "dashscope/qwen3.5-omni-plus",
            "dashscope/qwen3.5-omni-flash-2026-03-15",
            "gemini-3.8-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "custom-model",
        ):
            with self.subTest(model=model):
                args = argparse.Namespace(
                    model=model,
                    thinking=None,
                    media_input_type=None,
                    omit_modalities=None,
                )
                apply_model_defaults(args)
                gemini = model.startswith("gemini-")
                self.assertEqual(
                    args.media_input_type, "file" if gemini else "video_url"
                )
                self.assertEqual(args.omit_modalities, gemini)
                self.assertEqual(
                    args.thinking,
                    "off" if model.startswith("dashscope/") else "default",
                )
                args.thinking = "on"
                args.media_input_type = "video_url"
                args.omit_modalities = False
                apply_model_defaults(args)
                self.assertEqual(args.thinking, "on")
                self.assertEqual(args.media_input_type, "video_url")
                self.assertFalse(args.omit_modalities)

    def test_file_input_preserves_mp4_payload(self):
        part = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}}
        self.assertIs(media_input(part, "video_url"), part)
        self.assertEqual(
            media_input(part, "file"),
            {
                "type": "file",
                "file": {"file_data": part["video_url"]["url"], "filename": "clip.mp4"},
            },
        )

    def test_time_formats_and_rejection(self):
        self.assertEqual(seconds("01:02:50"), 62.5)
        self.assertEqual(who_cutoff("What happened from 0：04 to 0:08 seconds?"), 8)
        self.assertEqual(
            who_cutoff("What happened at the 2th and 6th second of the video?"), 6
        )
        self.assertEqual(who_cutoff("Who was speaking at 0:05 in the video?"), 5)
        with self.assertRaises(ValueError):
            who_cutoff("Who speaks sometime later?")

    def test_failures_remain_in_denominator(self):
        result = summarize(
            [
                {"task": "when", "gold": "YES", "prediction": "YES"},
                {
                    "task": "when",
                    "gold": "YES",
                    "prediction": None,
                    "error": "request failed",
                },
                {"task": "when", "gold": "NO", "prediction": None},
                {"task": "when", "gold": "NO", "prediction": "NO"},
            ]
        )["when"]
        self.assertEqual(result["accuracy"], 50)
        self.assertEqual(result["request_failures"], 1)
        self.assertEqual(result["parse_failures"], 1)
        self.assertEqual(result["class_counts"]["YES"]["fn"], 1)

    def test_all_paper_queries_are_identified(self):
        root = (
            Path(__file__).resolve().parents[2]
            / "reproducibility/arxiv-v3/final_source/data"
        )
        if not root.exists():
            self.fail("Missing archived paper annotations")
        l1 = json.loads((root / "level_1/dataset.json").read_text())
        l2 = json.loads((root / "level_2/annotations_200.json").read_text())["data"]
        self.assertEqual(len(l1), 2000)
        self.assertTrue(all(who_cutoff(row["question"]) > 0 for row in l1))
        self.assertEqual(sum(gold_when(row) == "YES" for row in l2), 128)
        self.assertTrue(all(participant(row) for row in l2))


if __name__ == "__main__":
    unittest.main()
