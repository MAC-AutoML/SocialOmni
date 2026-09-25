"""Serve MiniCPM-o 4.5 through the SocialOmni local AV adapter.

The optional MiniCPM-o runtime is intentionally imported inside ``load_model``.
The main SocialOmni environment can therefore keep its pinned Qwen runtime;
the server should be started from a separate Python 3.12 environment containing
``minicpmo-utils`` and the MiniCPM-o compatible Transformers release.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG  # noqa: E402
from models.model_server.local_common.gpu_visibility import (  # noqa: E402
    configure_cuda_visible_devices,
)

app = Flask(__name__)
PHYSICAL_GPUS = configure_cuda_visible_devices(
    CONFIG.model("minicpmo_4_5").get("gpu_ids", [])
    or CONFIG.runtime("gpu_ids", [])
)
MODEL_PATH = os.getenv("MINICPMO_4_5_MODEL_PATH") or CONFIG.model(
    "minicpmo_4_5"
).get("model_path")
MAX_TOKENS = int(CONFIG.model("minicpmo_4_5").get("max_tokens", 256))
model = None
get_video_frame_audio_segments = None


def load_model() -> None:
    global model, get_video_frame_audio_segments
    import torch
    from minicpmo.utils import get_video_frame_audio_segments as segmenter
    from transformers import AutoModel

    if not MODEL_PATH:
        raise RuntimeError("MiniCPM-o 4.5 model_path is not configured")
    get_video_frame_audio_segments = segmenter
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    ).eval().cuda()


def infer(video_path: str, question: str) -> str:
    import torch

    frames, audio_segments, stacked = get_video_frame_audio_segments(
        video_path, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
    )
    contents = []
    for index, frame in enumerate(frames):
        contents.append(frame)
        if index < len(audio_segments) and audio_segments[index] is not None:
            contents.append(audio_segments[index])
        if stacked is not None and index < len(stacked) and stacked[index] is not None:
            contents.append(stacked[index])
    messages = [
        model.get_sys_prompt(mode="omni", language="en"),
        {"role": "user", "content": contents + [question]},
    ]
    with torch.inference_mode():
        answer = model.chat(
            msgs=messages,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=0.0,
            use_tts_template=False,
            enable_thinking=False,
            omni_mode=True,
            generate_audio=False,
            max_slice_nums=1,
        )
    if isinstance(answer, tuple):
        answer = answer[0]
    return str(answer or "").strip()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/analyze")
def analyze():
    upload = request.files.get("video")
    question = request.form.get("question", "").strip()
    if upload is None or not upload.filename:
        return jsonify({"status": "error", "error": "video is required"}), 400
    if not question:
        return jsonify({"status": "error", "error": "question is required"}), 400
    temp_dir = Path(tempfile.mkdtemp(prefix="socialomni_minicpmo45_"))
    video_path = temp_dir / Path(upload.filename).name
    try:
        upload.save(video_path)
        return jsonify({"status": "success", "answer": infer(str(video_path), question)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=CONFIG.model("minicpmo_4_5").get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(CONFIG.model("minicpmo_4_5").get("port", 5096)))
    args = parser.parse_args()
    load_model()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
