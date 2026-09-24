"""Rescore fixed SocialOmni answers without regenerating candidate responses."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from client import StreamingClient, atomic_json, digest
from judge_eval import PANELS, RUBRIC, prompt

PANEL = "modern-20260924"
VALUES = {0, 25, 50, 75, 100}
PARAMETERS = {"temperature": 0, "top_p": 1, "max_tokens": 8192}


def record_key(row):
    return digest([row["model"], row["item_id"]])


def validate(data):
    if data.get("schema_version") != 1:
        raise ValueError("Expected normalized schema version 1")
    models = {}
    for model in data["models"]:
        name = model["model"]
        if not isinstance(name, str) or not name or name in models:
            raise ValueError("Missing or duplicate model ID")
        if not isinstance(model.get("full_gold"), bool):
            raise ValueError("full_gold must be explicit")
        rows = model["when"]
        by_id = {r["item_id"]: r for r in rows}
        gold = {r["item_id"] for r in rows if r["gold"] == "YES"}
        if len(rows) != 200 or len(by_id) != 200 or len(gold) != 128:
            raise ValueError("Expected 200 unique When items and 128 gold positives")
        if any(
            not isinstance(r["item_id"], str)
            or not r["item_id"]
            or r["gold"] not in {"YES", "NO"}
            or r.get("prediction") not in {"YES", "NO", None}
            for r in rows
        ):
            raise ValueError("Invalid When ID or label")
        expected = model.get(
            "expected_response_ids", sorted(gold) if model["full_gold"] else None
        )
        if not isinstance(expected, list) or len(set(expected)) != len(expected):
            raise ValueError("Explicit unique legacy response IDs are required")
        if not set(expected) <= gold or (model["full_gold"] and set(expected) != gold):
            raise ValueError(
                "Response IDs must match the declared gold-positive protocol"
            )
        if not model["full_gold"] and any(
            by_id[i].get("prediction") != "YES" for i in expected
        ):
            raise ValueError("Legacy responses must be predicted YES")
        models[name] = (model, by_id, set(expected))
    if not models:
        raise ValueError("At least one model is required")
    observed = {name: set() for name in models}
    for row in data["records"]:
        name, item = row["model"], row["item_id"]
        if name not in models or item not in models[name][2] or item in observed[name]:
            raise ValueError("Unknown, unexpected, or duplicate candidate ID")
        if row.get("error"):
            raise ValueError("Resolve candidate generation failures before scoring")
        if any(
            not isinstance(row.get(k), str)
            for k in ("context", "target_question", "reference", "candidate")
        ):
            raise ValueError(
                "Context, instruction, reference and candidate must be text"
            )
        observed[name].add(item)
    if any(observed[name] != expected for name, (_, _, expected) in models.items()):
        raise ValueError("Candidate IDs do not match expected response IDs")
    return models


def summarize(data, scores, panel):
    models = validate(data)
    summaries = []
    for name, (model, when, _) in models.items():
        rows = [r for r in data["records"] if r["model"] == name]
        covered = sum(
            bool(r["candidate"].strip())
            and when[r["item_id"]].get("prediction") == "YES"
            for r in rows
        )
        total = conditional = 0
        missing = []
        for row in rows:
            if not row["candidate"].strip():
                continue
            votes = scores.get(record_key(row), {})
            if set(votes) != panel or any(
                type(v.get("score")) is not int or v["score"] not in VALUES
                for v in votes.values()
            ):
                missing.append(row["item_id"])
                continue
            quality = sum(v["score"] for v in votes.values()) / len(panel)
            total += quality
            if when[row["item_id"]].get("prediction") == "YES":
                conditional += quality
        when_errors = [r["item_id"] for r in when.values() if r.get("error")]
        complete = not missing and not when_errors
        summaries.append(
            {
                "model": name,
                "source": model.get("source", data.get("source")),
                "full_gold": model["full_gold"],
                "complete": complete,
                "response_count": len(rows),
                "gold_positive": 128,
                "covered_positive": covered,
                "missing_score_ids": missing,
                "when_error_ids": when_errors,
                "who": model.get("who"),
                "when_summary": model.get("when_summary"),
                "qgold": total / 128 if complete and model["full_gold"] else None,
                "qens": conditional / covered if complete and covered else None,
                "cov_plus": 100 * covered / 128,
                "qens_joint": conditional / 128 if complete else None,
            }
        )
    return summaries


def public_report(data, scores, identity):
    models = summarize(data, scores, PANELS[identity["panel"]])
    records = []
    for row in data["records"]:
        records.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "model",
                        "item_id",
                        "context",
                        "target_question",
                        "reference",
                        "candidate",
                    )
                },
                "candidate_sha256": digest(row["candidate"]),
                "prompt_sha256": digest(prompt(row, row["candidate"].strip())),
                "scores": scores.get(record_key(row), {}),
            }
        )
    return {
        "schema_version": 1,
        "complete": all(m["complete"] for m in models),
        "provenance": identity,
        "models": models,
        "records": records,
    }


async def run(args):
    data = json.loads(args.input.read_text())
    validate(data)
    judges = json.loads(args.judges.read_text())["judges"]
    if len(judges) != 3 or {j["name"] for j in judges} != PANELS[args.panel]:
        raise ValueError("Judge configuration must match the selected panel")
    # Endpoint identities are hashed; the published result does not expose routing.
    identity = {
        "panel": args.panel,
        "source": data.get("source"),
        "input_sha256": digest(data),
        "input_file_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rubric": RUBRIC,
        "rubric_source": "Reconstructed from arXiv v3 Appendix A.6; not the original verbatim API prompt",
        "prompt_implementation_sha256": digest(
            (Path(__file__).parent / "judge_eval.py").read_text()
        ),
        "implementation_sha256": digest(Path(__file__).read_text()),
        "client_implementation_sha256": digest(
            (Path(__file__).parent / "client.py").read_text()
        ),
        "parameters": PARAMETERS,
        "judges": [
            {
                "name": j["name"],
                "model": j["model"],
                "include_modalities": j.get("include_modalities", True),
                "enable_thinking": j.get("enable_thinking"),
                "reasoning_effort": j.get("reasoning_effort"),
                "spec_sha256": digest(
                    {k: v for k, v in j.items() if k != "api_key_env"}
                ),
            }
            for j in judges
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(
            "Inputs or implementation changed; choose a new output directory"
        )
    atomic_json(manifest, identity)
    score_path = args.output / "scores.json"
    # Rebuild from exact-payload response caches, never from copied score totals.
    scores = {}
    errors = []
    completed = 0

    async def judge(spec):
        key = os.environ[spec["api_key_env"]] if spec.get("api_key_env") else "EMPTY"
        client = StreamingClient(
            key,
            args.output / spec["name"],
            base_url=spec["base_url"],
            model=spec["model"],
            timeout=600,
            trust_env=True,
        )
        semaphore = asyncio.Semaphore(spec.get("max_concurrency", 2))

        async def score(row):
            nonlocal completed
            if not row["candidate"].strip():
                return
            row_key = record_key(row)
            async with semaphore:
                try:
                    response = await client.complete(
                        [
                            {
                                "role": "user",
                                "content": prompt(row, row["candidate"].strip()),
                            }
                        ],
                        **PARAMETERS,
                        require_text=True,
                        include_modalities=spec.get("include_modalities", True),
                        enable_thinking=spec.get("enable_thinking"),
                        reasoning_effort=spec.get("reasoning_effort"),
                    )
                    value = response["text"].strip()
                    if value not in {str(v) for v in VALUES}:
                        raise ValueError(
                            "Judge did not return a permitted discrete score"
                        )
                    scores.setdefault(row_key, {})[spec["name"]] = {
                        "score": int(value),
                        "request_sha256": response["request_sha256"],
                        "payload_sha256": response["payload_sha256"],
                    }
                    atomic_json(score_path, scores)
                    completed += 1
                    if completed % 25 == 0:
                        atomic_json(
                            args.output / "results.json",
                            public_report(data, scores, identity),
                        )
                    print(
                        json.dumps(
                            {
                                "judge": spec["name"],
                                "model": row["model"],
                                "item": row["item_id"],
                                "score": int(value),
                            }
                        ),
                        flush=True,
                    )
                except (RuntimeError, ValueError, KeyError) as error:
                    errors.append(
                        {
                            "judge": spec["name"],
                            "model": row["model"],
                            "item": row["item_id"],
                            "error": type(error).__name__,
                        }
                    )

        await asyncio.gather(*(score(row) for row in data["records"]))

    await asyncio.gather(*(judge(spec) for spec in judges))
    report = public_report(data, scores, identity)
    report["judge_errors"] = errors
    atomic_json(args.output / "results.json", report)
    print(
        json.dumps(
            {"complete": report["complete"], "models": report["models"]}, indent=2
        )
    )
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=sorted(PANELS), default=PANEL)
    asyncio.run(run(parser.parse_args()))
