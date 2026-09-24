"""Candidate generation using the SocialOmni arXiv v3 prompt cards and item lists."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path

from client import StreamingClient, atomic_json, digest, load_key, video_part

WHO = "Identify who says what in the marked interval. Answer only A, B, C, or D."
WHEN = "You are an external dialogue observer. The target participant is {target}. Based only on the video and audio observed up to the current timestamp, should this participant begin a substantive speaking turn now? A means YES and B means NO. Answer only A or B."
HOW = "Write the next utterance that the target participant should say based only on the observed audio-visual prefix. Do not write speech for another participant. Output only the utterance."
ENCODING = {
    "video": "libx264",
    "crf": 18,
    "audio": "aac",
    "audio_bitrate": "192k",
    "timing": "trim-video-and-audio-before-encode-v1",
}


MODEL_DEFAULTS = {
    "dashscope/qwen3.8-omni-flash": ("off", "video_url", False),
    "dashscope/qwen3.5-omni-plus": ("off", "video_url", False),
    "dashscope/qwen3.5-omni-flash-2026-03-15": ("off", "video_url", False),
    "gemini-3.8-flash": ("default", "file", True),
}


def apply_model_defaults(args):
    thinking, media_type, omit_modalities = MODEL_DEFAULTS.get(
        args.model, ("default", "video_url", False)
    )
    if args.thinking is None:
        args.thinking = thinking
    if args.media_input_type is None:
        args.media_input_type = media_type
    if args.omit_modalities is None:
        args.omit_modalities = omit_modalities
    return args


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seconds(raw):
    text = str(raw).replace("：", ":")
    parts = text.split(":")
    if len(parts) == 1:
        result = float(parts[0])
    elif len(parts) == 2:
        result = int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3 and re.fullmatch(r"\d+:\d{2}:\d{2}", text):
        result = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100
    else:
        raise ValueError(f"Invalid timestamp: {raw}")
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid timestamp: {raw}")
    return result


def who_cutoff(question):
    question = question.replace("：", ":")
    patterns = [
        r"from\s+(\d+:\d+(?::\d+)?(?:\.\d+)?)\s+to\s+(\d+:\d+(?::\d+)?(?:\.\d+)?)",
        r"(?:from|between|at)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s*(?:seconds?|s)?\s+(?:to|and)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s*(?:seconds?|s)",
        r"between second (\d+) and (\d+)",
        r"at (\d+:\d+)",
    ]
    matches = [
        match for pattern in patterns for match in re.finditer(pattern, question, re.I)
    ]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous query interval: {question}")
    values = [seconds(value) for value in matches[0].groups()]
    if values[-1] <= 0 or (len(values) == 2 and values[0] > values[1]):
        raise ValueError(f"Invalid query interval: {question}")
    return values[-1]


def participant(row):
    question = row["question_2"]["question"].strip()
    match = re.fullmatch(r"What should (.+?) say\?", question, re.I)
    if not match:
        raise ValueError(f"Ambiguous participant: {question}")
    return match.group(1)


def gold_when(row):
    q = row["question_1"]
    answer = q["correct_answer"].upper()
    return q[f"option_{answer}"].upper() if answer in {"A", "B"} else answer


def parse_prediction(task, output):
    text = output.strip().upper()
    if task == "who":
        return text if text in {"A", "B", "C", "D"} else None
    if task == "when":
        return {"A": "YES", "B": "NO", "YES": "YES", "NO": "NO"}.get(text)
    return output.strip() or None


def media_input(part, input_type):
    if input_type == "file":
        return {
            "type": "file",
            "file": {"file_data": part["video_url"]["url"], "filename": "clip.mp4"},
        }
    return part


def media_path(root, level, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe media path")
    candidates = [
        root / "data" / level / "videos" / relative,
        root / level / "videos" / relative,
        root / "data" / level / relative,
        root / level / relative,
        root / relative,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
            return candidate
    raise FileNotFoundError(name)


async def prefix(source, cutoff, directory, source_hash=None):
    source_hash = source_hash or await asyncio.to_thread(file_hash, source)
    identity = {
        "source_sha256": source_hash,
        "cutoff_seconds": cutoff,
        "encoding": ENCODING,
    }
    path = directory / (digest(identity) + ".mp4")
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata.get("identity") == identity and metadata.get(
            "output_sha256"
        ) == await asyncio.to_thread(file_hash, path):
            return path, identity
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(source),
        "-t",
        str(cutoff),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"trim=end={cutoff},setpts=PTS-STARTPTS",
        "-af",
        f"atrim=end={cutoff},asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError("ffmpeg prefix failed: " + stderr.decode()[:300])
    temporary.replace(path)
    atomic_json(
        sidecar,
        {
            "identity": identity,
            "output_sha256": await asyncio.to_thread(file_hash, path),
        },
    )
    return path, identity


def summarize(rows):
    summary = {}
    for task, labels in [("who", list("ABCD")), ("when", ["YES", "NO"])]:
        items = [row for row in rows if row["task"] == task]
        if not items:
            continue
        correct = sum(row.get("prediction") == row["gold"] for row in items)
        f1s = []
        counts = {}
        for label in labels:
            tp = sum(
                row["gold"] == label and row.get("prediction") == label for row in items
            )
            fp = sum(
                row["gold"] != label and row.get("prediction") == label for row in items
            )
            fn = sum(
                row["gold"] == label and row.get("prediction") != label for row in items
            )
            counts[label] = {"tp": tp, "fp": fp, "fn": fn}
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        summary[task] = {
            "n": len(items),
            "correct": correct,
            "accuracy": 100 * correct / len(items),
            "macro_f1": 100 * sum(f1s) / len(f1s),
            "class_counts": counts,
            "request_failures": sum(bool(row.get("error")) for row in items),
            "parse_failures": sum(
                not row.get("error") and row.get("prediction") is None for row in items
            ),
        }
    summary["how"] = {"n": sum(row["task"] == "how" for row in rows), "judged": False}
    return summary


async def run(args):
    l1path = args.annotations / "level_1/dataset.json"
    l2path = args.annotations / "level_2/annotations_200.json"
    l1 = json.loads(l1path.read_text())
    raw = json.loads(l2path.read_text())
    l2 = raw["data"] if isinstance(raw, dict) else raw
    if (len(l1), len(l2), sum(gold_when(row) == "YES" for row in l2)) != (
        2000,
        200,
        128,
    ):
        raise ValueError("Unexpected v3 dataset counts")
    # Validate every query before making any billed request.
    cutoffs = {str(row["id"]): who_cutoff(row["question"]) for row in l1}
    for row in l2:
        participant(row)
        seconds(row["question_1"]["timestamp"])
    selected = {}
    inventory = {}
    paths = {}
    for task in args.tasks:
        selection = (
            l1
            if task == "who"
            else [row for row in l2 if task != "how" or gold_when(row) == "YES"]
        )
        selected[task] = selection[: args.limit] if args.limit else selection
        for row in selected[task]:
            sample_id = str(row["id"] if task == "who" else row["video_id"])
            level = "level_1" if task == "who" else "level_2"
            source = media_path(
                args.media,
                level,
                row["video_path"] if task == "who" else row["video_file"],
            )
            paths[(task, sample_id)] = source
            inventory[f"{task}-{sample_id}"] = await asyncio.to_thread(
                file_hash, source
            )
    thinking = {"default": None, "off": False, "on": True}[args.thinking]
    identity = {
        "protocol": "socialomni-arxiv-v3-a8-a10-prefix-v1",
        "endpoint": args.base_url,
        "source_media_sha256": inventory,
        "source_sha256": {"level1": file_hash(l1path), "level2": file_hash(l2path)},
        "prompts": {"who": WHO, "when": WHEN, "how": HOW},
        "model": args.model,
        "temperature": 0.3,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "enable_thinking": thinking,
        "include_modalities": not args.omit_modalities,
        "media_input_type": args.media_input_type,
        "prefix_encoding": ENCODING,
        "who_cutoff_rule": "interval-end-or-explicit-point; paired-points-use-later; normalize-fullwidth-colon",
        "tasks": args.tasks,
        "limit": args.limit,
        "harness_sha256": {
            name: file_hash(Path(__file__).parent / name)
            for name in ["evaluate.py", "client.py"]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Output manifest differs; choose a new output directory")
    atomic_json(manifest, identity)
    manifest_hash = digest(identity)
    key = load_key(args.key_file, args.key_field)
    client = StreamingClient(
        key,
        args.output / "requests",
        base_url=args.base_url,
        model=args.model,
        timeout=600,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    items_dir = args.output / "items"
    items_dir.mkdir(exist_ok=True)

    async def item(task, row):
        sample_id = str(row["id"] if task == "who" else row["video_id"])
        saved = items_dir / f"{task}-{sample_id}.json"
        if saved.exists():
            previous = json.loads(saved.read_text())
            if (
                previous.get("manifest_sha256") != manifest_hash
                or previous.get("source_media_sha256")
                != inventory[f"{task}-{sample_id}"]
            ):
                raise ValueError("Cached item identity differs")
            if not previous.get("error"):
                rows.append(previous)
                return
        async with semaphore:
            result = {
                "task": task,
                "sample_id": sample_id,
                "manifest_sha256": manifest_hash,
                "source_media_sha256": inventory[f"{task}-{sample_id}"],
                "prediction": None,
                "gold": row["correct_answer"] if task == "who" else gold_when(row),
            }
            try:
                source = paths[(task, sample_id)]
                cutoff = (
                    cutoffs[sample_id]
                    if task == "who"
                    else seconds(row["question_1"]["timestamp"])
                )
                video, provenance = await prefix(
                    source,
                    cutoff,
                    args.output / "prefixes",
                    inventory[f"{task}-{sample_id}"],
                )
                part, media_hash = await asyncio.to_thread(video_part, video)
                part = media_input(part, args.media_input_type)
                if task == "who":
                    prompt = (
                        WHO + "\n" + row["question"] + "\n" + "\n".join(row["options"])
                    )
                elif task == "when":
                    prompt = WHEN.format(target=participant(row))
                else:
                    prompt = HOW + "\nTarget participant: " + participant(row)
                response = await client.complete(
                    [
                        {
                            "role": "user",
                            "content": [part, {"type": "text", "text": prompt}],
                        }
                    ],
                    media_sha256=[media_hash],
                    max_tokens=args.max_tokens,
                    enable_thinking=thinking,
                    include_modalities=not args.omit_modalities,
                )
                result.update(
                    text=response["text"],
                    prediction=parse_prediction(task, response["text"]),
                    request_sha256=response["request_sha256"],
                    prefix=provenance,
                    media_sha256=media_hash,
                    finish_reason=response["finish_reason"],
                    usage=response["usage"],
                )
            except (OSError, ValueError, RuntimeError) as error:
                result["error"] = (
                    type(error).__name__ + ": " + str(error).replace(key, "[REDACTED]")
                )
            atomic_json(saved, result)
            rows.append(result)
            print(
                json.dumps(
                    {
                        "task": task,
                        "id": sample_id,
                        "error": result.get("error"),
                        "prediction": result["prediction"]
                        if task != "how"
                        else bool(result["prediction"]),
                    }
                ),
                flush=True,
            )

    for task in args.tasks:
        selection = selected[task]
        await asyncio.gather(*(item(task, row) for row in selection))
        atomic_json(args.output / "summary.json", summarize(rows))
    atomic_json(args.output / "candidates.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path, required=True, help="v3 final_source/data directory"
    )
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--key-field", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--model", default="dashscope/qwen3.8-omni-flash")
    parser.add_argument("--thinking", choices=["default", "on", "off"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--media-input-type", choices=["video_url", "file"])
    parser.add_argument(
        "--omit-modalities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Omit the optional output modalities field for providers that reject it",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["who", "when", "how"],
        default=["when", "how", "who"],
    )
    parser.add_argument("--limit", type=int)
    args = apply_model_defaults(parser.parse_args())
    if not args.base_url:
        parser.error("Set --base-url or OPENAI_API_BASE")
    if (
        args.concurrency < 1
        or args.max_tokens < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Counts must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
