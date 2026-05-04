from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config.paths import PATHS
from config.settings import CONFIG


def modality_metadata(level: int) -> dict[str, Any]:
    raw = str(
        os.getenv(f"SOCIALOMNI_LEVEL{level}_MODALITY")
        or os.getenv("SOCIALOMNI_MODALITY")
        or os.getenv("MODALITY")
        or CONFIG.benchmark(f"level{level}.modality", "avt")
    ).strip().lower()
    if raw in {"vt", "v", "vision", "vision+text", "video+text"}:
        return {
            "modality": "video-only",
            "modality_code": "vt",
            "use_video": True,
            "use_audio": False,
        }
    if raw in {"at", "a", "audio", "audio+text"}:
        return {
            "modality": "audio-only",
            "modality_code": "at",
            "use_video": False,
            "use_audio": True,
            "visual_mask": False,
        }
    if raw in {
        "amv",
        "audio-masked-video",
        "audio-only-masked-video",
        "masked-video-audio",
    }:
        return {
            "modality": "audio-only-masked-video",
            "modality_code": "amv",
            "use_video": True,
            "use_audio": True,
            "visual_mask": True,
        }
    return {
        "modality": "audio-video",
        "modality_code": "avt",
        "use_video": True,
        "use_audio": True,
        "visual_mask": False,
    }


def add_payload_modality(payload: dict[str, Any], level: int) -> dict[str, Any]:
    meta = modality_metadata(level)
    payload.update(meta)
    payload["experiment"] = {
        "level": level,
        **meta,
        "note": _experiment_note(meta),
    }
    return payload


def add_row_modality(row: dict[str, Any], level: int) -> dict[str, Any]:
    row.update(modality_metadata(level))
    return row


def output_path_for(level: int, model_name: str) -> Path:
    output_dir = CONFIG.benchmark(f"level{level}.output_dir", "")
    output_pattern = CONFIG.benchmark(
        f"level{level}.output_pattern",
        f"results_{{model}}_level{level}_{{modality}}.json",
    )
    meta = modality_metadata(level)
    output_base = Path(output_dir) if output_dir else PATHS.results_dir
    return output_base / output_pattern.format(model=model_name, **meta)


def _experiment_note(meta: dict[str, Any]) -> str:
    if meta["modality_code"] == "at":
        return "Audio-only evaluation: video file is used only as the audio source; visual frames are not passed to the model."
    if meta["modality_code"] == "vt":
        return "Video-only evaluation: visual frames are passed to the model; audio is not passed."
    if meta["modality_code"] == "amv":
        return "Audio-only masked-video evaluation: black visual frames are passed with the original audio to keep video-token pressure comparable."
    return "Audio-video evaluation: both visual frames and audio are passed to the model."
