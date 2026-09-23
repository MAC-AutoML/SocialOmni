"""Export per-item predictions and complete-panel metrics without private endpoints."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate import summarize
from judge_eval import JUDGES, metrics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(candidates, judges, output, allow_incomplete=False):
    manifest_path = candidates / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [
        json.loads(p.read_text()) for p in sorted((candidates / "items").glob("*.json"))
    ]
    classification = summarize(rows)
    if (
        classification.get("who", {}).get("n") != 2000
        or classification.get("when", {}).get("n") != 200
    ):
        raise ValueError("Complete Who and When sample lists are required")
    when = [r for r in rows if r["task"] == "when"]
    how = [r for r in rows if r["task"] == "how"]
    score_path = judges / "scores.json"
    scores = json.loads(score_path.read_text()) if score_path.exists() else {}
    quality = metrics(when, how, scores)
    when_by_id = {r["sample_id"]: r for r in when}
    quality["covered_positive"] = sum(
        not r.get("error")
        and bool((r.get("text") or "").strip())
        and when_by_id[r["sample_id"]].get("prediction") == "YES"
        for r in how
    )
    quality["cov_plus"] = 100 * quality["covered_positive"] / 128
    complete = quality["complete"] and not any(r.get("error") for r in rows)
    if not complete and not allow_incomplete:
        raise ValueError("Evaluation is incomplete; no official result can be exported")
    judge_manifest = judges / "manifest.json"
    records = []
    for row in rows:
        record = {
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
        }
        record["request_success"] = not bool(row.get("error"))
        if row["task"] == "how":
            record["judge_scores"] = scores.get(row["sample_id"], {})
        records.append(record)
    result = {
        "model": manifest["model"],
        "status": "complete" if complete else "incomplete",
        "paper": "https://arxiv.org/abs/2603.16859v3",
        "protocol": manifest["protocol"],
        "scope": "Supplemental hosted evaluation; original primary API envelopes were not fully released",
        "inference": {
            k: manifest[k]
            for k in (
                "temperature",
                "top_p",
                "max_tokens",
                "enable_thinking",
                "prefix_encoding",
                "who_cutoff_rule",
            )
        },
        "provenance": {
            "candidate_manifest_sha256": sha256(manifest_path),
            "judge_manifest_sha256": sha256(judge_manifest)
            if judge_manifest.exists()
            else None,
            "metadata_sha256": manifest["source_sha256"],
            "candidate_harness_sha256": manifest["harness_sha256"],
            "endpoints": "Private relay addresses and credentials omitted",
        },
        "classification": {k: classification[k] for k in ("who", "when")},
        "generation": {
            "count": len(how),
            "nonempty": sum(bool((r.get("text") or "").strip()) for r in how),
        },
        "judges": sorted(JUDGES),
        "judge_score_count": sum(len(v) for v in scores.values()),
        "quality": quality,
        "metrics": {
            "who": classification["who"]["accuracy"],
            "when": classification["when"]["accuracy"],
            **{k: quality[k] for k in ("qgold", "qens", "cov_plus", "qens_joint")},
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("status", "metrics", "judge_score_count")}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    export(args.candidates, args.judges, args.output, args.allow_incomplete)
