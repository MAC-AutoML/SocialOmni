from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from config.paths import PATHS

DATASET_REPO_ID = "alexisty/SocialOmni"


@dataclass(frozen=True)
class DatasetSpec:
    level_key: str
    dataset_relpath: str
    video_dir_relpath: str
    allow_patterns: tuple[str, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "level1": DatasetSpec(
        level_key="level1",
        dataset_relpath="data/level_1/dataset.json",
        video_dir_relpath="data/level_1/videos",
        allow_patterns=("data/level_1/**",),
    ),
    "level2": DatasetSpec(
        level_key="level2",
        dataset_relpath="data/level_2/annotations.json",
        video_dir_relpath="data/level_2/videos",
        allow_patterns=("data/level_2/**",),
    ),
}


def _auto_download_enabled() -> bool:
    raw = os.getenv("SOCIALOMNI_AUTO_DOWNLOAD_DATASET", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _required_video_paths(level_key: str, dataset_path: Path) -> list[str]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if level_key == "level1":
        if not isinstance(payload, list):
            return []
        return [str(item.get("video_path", "")).strip() for item in payload if item.get("video_path")]

    if level_key == "level2":
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        if not isinstance(payload, list):
            return []
        return [str(item.get("video_file", "")).strip() for item in payload if item.get("video_file")]

    return []


def dataset_is_ready(level_key: str, dataset_path: Path, video_dir: Path) -> bool:
    if not dataset_path.exists() or not video_dir.exists() or not video_dir.is_dir():
        return False

    required = _required_video_paths(level_key, dataset_path)
    if not required:
        return False

    for relpath in required:
        if not (video_dir / relpath).exists():
            return False
    return True


def ensure_default_dataset_available(level_key: str, dataset_path: Path, video_dir: Path) -> bool:
    spec = DATASET_SPECS.get(level_key)
    if spec is None:
        return False

    if dataset_is_ready(level_key, dataset_path, video_dir):
        return False

    if not _auto_download_enabled():
        return False

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Dataset is missing and auto-download requires huggingface_hub in the project environment."
        ) from exc

    print(
        f"[INFO] Missing default {level_key} dataset under {PATHS.data_dir}. "
        f"Downloading from Hugging Face dataset {DATASET_REPO_ID}...",
        flush=True,
    )
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=str(PATHS.root),
        allow_patterns=list(spec.allow_patterns),
    )

    if not dataset_is_ready(level_key, dataset_path, video_dir):
        raise RuntimeError(
            f"Dataset download completed but expected files are still missing: "
            f"dataset={dataset_path}, videos={video_dir}"
        )
    return True
