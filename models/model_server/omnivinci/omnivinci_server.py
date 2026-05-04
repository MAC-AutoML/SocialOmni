import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG
from models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices
from models.model_server.local_common.transformers_compat import (
    ensure_all_tied_weights_keys,
    ensure_transformers_no_init_weights,
)

warnings.filterwarnings("ignore")

app = Flask(__name__)


# IMPORTANT: GPU visibility must be set before importing transformers
SPECIFIED_GPUS = configure_cuda_visible_devices(
    CONFIG.model("omnivinci").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

# Global configuration
MODEL_PATH = CONFIG.model("omnivinci").get("model_path") or "/publicssd/xty/models/omnivinci"
LOAD_AUDIO_IN_VIDEO = CONFIG.model("omnivinci").get("use_audio_in_video", True)
NUM_VIDEO_FRAMES = CONFIG.model("omnivinci").get("num_video_frames", 256)
AUDIO_LENGTH = "max_7200"
MAX_NEW_TOKENS = CONFIG.model("omnivinci").get("max_new_tokens", 4096)

# Global runtime objects
model = None
processor = None
config = None
generation_config = None
model_loaded = False


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def load_model():
    global model, processor, config, generation_config, model_loaded

    if model_loaded:
        return

    from transformers import AutoConfig, AutoModel, AutoProcessor
    import torch

    print(f"Loading OmniVinci model to GPUs {os.environ.get('CUDA_VISIBLE_DEVICES')}...")

    ensure_transformers_no_init_weights()
    ensure_all_tied_weights_keys()

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    generation_config = model.default_generation_config
    generation_config.update(max_new_tokens=MAX_NEW_TOKENS)
    generation_config.max_length = None

    model.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO
    processor.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO

    if NUM_VIDEO_FRAMES > 0:
        model.config.num_video_frames = NUM_VIDEO_FRAMES
        processor.config.num_video_frames = NUM_VIDEO_FRAMES

    if AUDIO_LENGTH != -1:
        model.config.audio_chunk_length = AUDIO_LENGTH
        processor.config.audio_chunk_length = AUDIO_LENGTH

    model_loaded = True
    print("Model loaded successfully.")


def _build_conversation(video_path: str, question: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": question},
            ],
        }
    ]


def run_inference(video_path: str, question: str, use_video: bool, use_audio: bool) -> str:
    assert model_loaded and model is not None and processor is not None, "Model is not loaded"

    model.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO and use_audio
    processor.config.load_audio_in_video = LOAD_AUDIO_IN_VIDEO and use_audio

    if use_video:
        model.config.num_video_frames = NUM_VIDEO_FRAMES
        processor.config.num_video_frames = NUM_VIDEO_FRAMES
    else:
        model.config.num_video_frames = 0
        processor.config.num_video_frames = 0

    conversation = _build_conversation(video_path, question)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor([text])

    media = inputs.media if hasattr(inputs, "media") else {}
    has_video = bool(media.get("video")) if isinstance(media, dict) else False
    has_audio = False
    if isinstance(media, dict):
        video_info = media.get("video_info")
        if isinstance(video_info, list) and video_info:
            first_info = video_info[0]
            if isinstance(first_info, dict) and "has_audio" in first_info:
                has_audio = bool(first_info.get("has_audio"))
        if not has_audio:
            has_audio = bool(media.get("sound")) or bool(media.get("audio_info"))
    print(f"Request media: video={has_video} audio={has_audio}")

    input_ids = inputs.input_ids
    if hasattr(inputs, "media"):
        media = inputs.media
    else:
        media = None

    if hasattr(inputs, "media_config"):
        media_config = inputs.media_config
    else:
        media_config = None

    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        generation_config=generation_config,
    )

    response = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response.strip()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "Video file not uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    question = request.form.get("question", "")
    use_video = _parse_bool(request.form.get("use_video"), True)
    use_audio = _parse_bool(request.form.get("use_audio"), LOAD_AUDIO_IN_VIDEO)
    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    temp_dir = None
    temp_path = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="omnivinci_server_")
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)

        answer = run_inference(temp_path, question, use_video, use_audio)
        return jsonify({"status": "success", "answer": answer.strip()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "error": str(exc)}), 500
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="OmniVinci Video Analysis Server")
    parser.add_argument("--port", type=int, default=5091, help="Server port (default: 5091)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model()
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
