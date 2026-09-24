"""Normalize archived answers and hosted runs for a disclosed judge panel."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluate import gold_when


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_input(root):
    annotations = json.loads(
        (root / "final_source/data/level_2/annotations_200.json").read_text()
    )["data"]
    gold = {row["video_id"]: gold_when(row) for row in annotations}
    answers_path = root / "results/primary/ensemble_response_scores.csv"
    decisions_path = root / "results/when_predictions.csv"
    answers = read_csv(answers_path)
    decisions = read_csv(decisions_path)
    summaries = read_csv(root / "results/primary/model_summary.csv")
    models = []
    records = []
    for summary in summaries:
        model = summary["model"]
        selected = [row for row in answers if row["model"] == model]
        when = []
        for row in decisions:
            if row["model"] != model:
                continue
            prediction = {"A": "YES", "B": "NO", "YES": "YES", "NO": "NO"}.get(
                row["q1_prediction"]
            )
            entry = {
                "item_id": row["item_id"],
                "gold": gold[row["item_id"]],
                "prediction": prediction,
            }
            if row["request_failed"].lower() == "true":
                entry["request_failed"] = True
            when.append(entry)
        accuracy = sum(row["gold"] == row["prediction"] for row in when) * 100 / 200
        if abs(accuracy - float(summary["when_accuracy"])) > 1e-9:
            raise ValueError(f"When accuracy mismatch for {model}")
        models.append(
            {
                "model": model,
                "display_model": summary["display_model"],
                "source": "arxiv-v3 canonical primary responses",
                "full_gold": False,
                "expected_response_ids": [row["item_id"] for row in selected],
                "who": {
                    "n": int(summary["who_n"]),
                    "accuracy": float(summary["who_accuracy"]),
                    "macro_f1": float(summary["who_macro_f1"]),
                },
                "when_summary": {
                    "n": 200,
                    "accuracy": accuracy,
                    "macro_f1": float(summary["when_macro_f1"]),
                },
                "when": when,
            }
        )
        for row in selected:
            records.append(
                {
                    "model": model,
                    "item_id": row["item_id"],
                    "archived_candidate_hash": row["candidate_hash"],
                    **{
                        k: row[k]
                        for k in (
                            "context",
                            "target_question",
                            "reference",
                            "candidate",
                        )
                    },
                }
            )
    return {
        "schema_version": 1,
        "source": "arXiv:2603.16859v3 primary answers; no new candidate generation",
        "source_sha256": {
            "answers": file_hash(answers_path),
            "decisions": file_hash(decisions_path),
        },
        "models": models,
        "records": records,
    }


def hosted_input(root, contexts_path, model):
    rows = [json.loads(p.read_text()) for p in sorted((root / "items").glob("*.json"))]
    contexts = json.loads(contexts_path.read_text())
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    when = [
        {
            "item_id": row["sample_id"],
            "gold": row["gold"],
            "prediction": row.get("prediction"),
            **({"request_failed": True} if row.get("error") else {}),
        }
        for row in rows
        if row["task"] == "when"
    ]
    records = []
    for row in rows:
        if row["task"] != "how":
            continue
        if row.get("error"):
            raise ValueError(
                "Resolve generation request failures before preparing scores"
            )
        records.append(
            {
                "model": model,
                "item_id": row["sample_id"],
                **contexts[row["sample_id"]],
                "candidate": row.get("text") or "",
            }
        )
    return {
        "schema_version": 1,
        "source": "Supplemental hosted candidate run",
        "source_sha256": {
            "candidate_manifest": file_hash(root / "manifest.json"),
            "contexts": file_hash(contexts_path),
        },
        "models": [
            {
                "model": model,
                "source": manifest["model"],
                "full_gold": True,
                "who": summary.get("who")
                if summary.get("who", {}).get("n") == 2000
                else None,
                "when_summary": summary["when"],
                "when": when,
            }
        ],
        "records": records,
    }


def candidate_export(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((root / "items").glob("*.json"))]
    from evaluate import summarize

    summary = summarize(rows)
    if (summary["who"]["n"], summary["when"]["n"], summary["how"]["n"]) != (
        2000,
        200,
        128,
    ):
        raise ValueError("Expected the complete 2000/200/128 candidate list")
    return {
        "manifest_sha256": file_hash(root / "manifest.json"),
        "configuration": {
            k: manifest[k]
            for k in (
                "model",
                "protocol",
                "temperature",
                "top_p",
                "max_tokens",
                "enable_thinking",
                "prefix_encoding",
                "who_cutoff_rule",
                "source_sha256",
                "harness_sha256",
                "include_modalities",
                "media_input_type",
                "prompts",
            )
            if k in manifest
        },
        "summary": summary,
        "records": [
            {
                **{
                    k: row.get(k)
                    for k in (
                        "task",
                        "sample_id",
                        "gold",
                        "prediction",
                        "text",
                        "finish_reason",
                        "source_media_sha256",
                        "media_sha256",
                        "request_sha256",
                    )
                },
                "request_success": not bool(row.get("error")),
            }
            for row in rows
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paper", type=Path)
    source.add_argument("--candidates", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    args = parser.parse_args()
    if args.candidates and (not args.contexts or not args.model):
        parser.error("--candidates requires --contexts and --model")
    data = (
        paper_input(args.paper)
        if args.paper
        else hosted_input(args.candidates, args.contexts, args.model)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if args.candidate_output:
        if not args.candidates:
            parser.error("--candidate-output requires --candidates")
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(candidate_export(args.candidates), ensure_ascii=False, indent=2)
            + "\n"
        )
    print(
        json.dumps({"models": len(data["models"]), "responses": len(data["records"])})
    )
