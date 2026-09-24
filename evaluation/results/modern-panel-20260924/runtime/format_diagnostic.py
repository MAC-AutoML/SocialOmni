"""Compare strict predictions with the archived repository choice extractor."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

MODELS = ("qwen38", "gemini38", "qwen35plus", "qwen35flash")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(path, extract_choice):
    data = json.loads(path.read_text())
    result = {"candidate_file_sha256": sha256(path), "tasks": {}, "records": []}
    for task, count, labels in (
        ("who", 2000, set("ABCD")),
        ("when", 200, {"YES", "NO"}),
    ):
        rows = [r for r in data["records"] if r["task"] == task]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count:
            raise ValueError(f"{path.parent.name}: expected {count} unique {task} rows")
        stats = {
            "n": count,
            "strict_correct": 0,
            "auxiliary_correct": 0,
            "eligible_format_failures": 0,
            "recovered_correct": 0,
            "recovered_incorrect": 0,
            "still_unparsed": 0,
            "request_failures": 0,
        }
        for row in rows:
            strict = row["prediction"]
            if row["gold"] not in labels or (
                strict is not None and strict not in labels
            ):
                raise ValueError(f"Invalid {task} prediction or gold label")
            if type(row.get("request_success")) is not bool:
                raise ValueError(
                    "Every candidate must have an explicit request-success flag"
                )
            auxiliary = strict
            eligible = row["request_success"] and strict is None
            if eligible:
                choice = extract_choice(
                    row["text"] or "", set("ABCD") if task == "who" else set("AB")
                )
                auxiliary = choice or None
                if task == "when":
                    auxiliary = {"A": "YES", "B": "NO"}.get(choice)
                stats["eligible_format_failures"] += 1
                if auxiliary is None:
                    stats["still_unparsed"] += 1
                else:
                    stats[
                        "recovered_correct"
                        if auxiliary == row["gold"]
                        else "recovered_incorrect"
                    ] += 1
            stats["request_failures"] += not row["request_success"]
            stats["strict_correct"] += strict == row["gold"]
            stats["auxiliary_correct"] += auxiliary == row["gold"]
            result["records"].append(
                {
                    "task": task,
                    "sample_id": row["sample_id"],
                    "gold": row["gold"],
                    "request_success": row["request_success"],
                    "strict_prediction": strict,
                    "auxiliary_prediction": auxiliary,
                    "extractor_applied": eligible,
                }
            )
        stats["strict_accuracy"] = 100 * stats["strict_correct"] / count
        stats["auxiliary_accuracy"] = 100 * stats["auxiliary_correct"] / count
        archived = data["summary"][task]
        if (
            stats["strict_correct"] != archived["correct"]
            or abs(stats["strict_accuracy"] - archived["accuracy"]) > 1e-9
        ):
            raise ValueError(f"{path.parent.name}: strict {task} summary mismatch")
        result["tasks"][task] = stats
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory containing all four model folders and candidates.json files",
    )
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.results / name / "candidates.json" for name in MODELS}
    missing = [str(p) for p in [args.extractor, *paths.values()] if not p.is_file()]
    if missing:
        parser.error("Required input files are missing: " + ", ".join(missing))
    if args.output.resolve() in {
        p.resolve() for p in [args.extractor, *paths.values()]
    }:
        parser.error("Output must not overwrite an input file")
    spec = importlib.util.spec_from_file_location(
        "socialomni_answer_extraction", args.extractor
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "schema_version": 1,
        "purpose": "Auxiliary format diagnostic; does not replace strict benchmark scores",
        "method": "Preserve every valid strict prediction. Apply the repository extractor only to successful requests with no strict prediction. Who uses A/B/C/D; When uses A/B mapped to YES/NO.",
        "limitations": "Permissive pattern extraction is not human adjudication. It may select an option mentioned in an explanation; recovered labels do not prove unambiguous intent. Failed requests remain in fixed denominators. This diagnostic does not recompute response quality or coverage.",
        "extractor_sha256": sha256(args.extractor),
        "diagnostic_implementation_sha256": sha256(Path(__file__)),
        "models": {
            name: diagnose(path, module.extract_choice) for name, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {name: value["tasks"] for name, value in report["models"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
