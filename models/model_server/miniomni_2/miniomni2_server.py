import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path
import subprocess

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG
from models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices

# Add local mini-omni2 library to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINI_OMNI2_LIB = os.path.join(SCRIPT_DIR, "mini_omni2_lib")
sys.path.insert(0, MINI_OMNI2_LIB)

warnings.filterwarnings("ignore")

app = Flask(__name__)

# IMPORTANT: GPU visibility must be set before importing torch and other models
PHYSICAL_GPUS = configure_cuda_visible_devices(
    CONFIG.model("miniomni_2").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

# Use logical GPU 0
LOGICAL_GPU = 0

# Global configuration
CHECKPOINT_PATH = CONFIG.model("miniomni_2").get("model_path") or "/publicssd/xty/models/mini-omni2"
MAX_RETURNED_TOKENS = CONFIG.model("miniomni_2").get("max_returned_tokens", 4096)
TEMPERATURE = CONFIG.model("miniomni_2").get("temperature", 0.9)
TOP_K = CONFIG.model("miniomni_2").get("top_k", 1)
TOP_P = CONFIG.model("miniomni_2").get("top_p", 1.0)
USE_AUDIO_IN_VIDEO = CONFIG.model("miniomni_2").get("use_audio_in_video", True)

# Global runtime objects
model_client = None
model_loaded = False
OmniVisionInference = None


def _parse_bool(value, default=True):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def create_silent_audio(output_path, duration=1.0, sample_rate=16000):
    import soundfile as sf
    import numpy as np

    num_samples = int(duration * sample_rate)
    silent_audio = np.zeros(num_samples, dtype=np.float32)
    sf.write(output_path, silent_audio, sample_rate)


def create_dummy_image(output_path, width=224, height=224):
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    img.save(output_path)


def extract_audio_from_video(video_path, output_audio_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error extracting audio: {exc.stderr}")
        return False


def extract_frame_from_video(video_path, output_image_path, timestamp=None):
    if timestamp is None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            duration = float(result.stdout.strip())
            timestamp = duration / 2
        except (subprocess.CalledProcessError, ValueError):
            timestamp = 1.0

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        str(timestamp),
        "-vframes",
        "1",
        str(output_image_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error extracting frame: {exc.stderr}")
        return False


def load_model():
    global model_client, model_loaded, OmniVisionInference

    if model_loaded:
        return

    from inference_vision_text import OmniVisionTextInference as _OmniVisionTextInference

    OmniVisionInference = _OmniVisionTextInference

    print(f"Loading Mini-Omni2 model to physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} (logical GPU {LOGICAL_GPU})...")
    device = f"cuda:{LOGICAL_GPU}"
    model_client = OmniVisionInference(ckpt_dir=CHECKPOINT_PATH, device=device)

    print("Warming up model...")
    temp_dir = "/tmp/miniomni2_warmup"
    os.makedirs(temp_dir, exist_ok=True)
    warm_up_audio = os.path.join(temp_dir, "warmup_audio.wav")
    warm_up_image = os.path.join(temp_dir, "warmup_image.jpg")
    create_silent_audio(warm_up_audio, duration=1.0)
    create_dummy_image(warm_up_image)

    try:
        for _ in model_client.run_vision_with_text_question(
            warm_up_audio,
            warm_up_image,
            "Describe this image.",
            save_path=None,
            warm_up=True,
        ):
            pass
    finally:
        if os.path.exists(warm_up_audio):
            os.remove(warm_up_audio)
        if os.path.exists(warm_up_image):
            os.remove(warm_up_image)

    model_loaded = True
    print("Model loaded and warmed up successfully.")


def run_inference(video_path, question, temp_dir, use_video, use_audio):
    assert model_loaded and model_client is not None, "Model is not loaded"

    video_stem = Path(video_path).stem
    temp_audio = os.path.join(temp_dir, f"{video_stem}_audio.wav")
    temp_image = os.path.join(temp_dir, f"{video_stem}_frame.jpg")

    if USE_AUDIO_IN_VIDEO and use_audio:
        if not extract_audio_from_video(video_path, temp_audio):
            raise RuntimeError(f"Failed to extract audio from {video_path}")
    else:
        create_silent_audio(temp_audio, duration=1.0)

    if use_video:
        if not extract_frame_from_video(video_path, temp_image, timestamp=None):
            raise RuntimeError(f"Failed to extract frame from {video_path}")
    else:
        create_dummy_image(temp_image)

    res_text = ""
    try:
        for _, text_stream in model_client.run_vision_with_text_question(
            temp_audio,
            temp_image,
            question,
            max_returned_tokens=MAX_RETURNED_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            save_path=None,
        ):
            res_text += text_stream
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        if os.path.exists(temp_image):
            os.remove(temp_image)

    return res_text.strip()


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
    use_audio = _parse_bool(request.form.get("use_audio"), USE_AUDIO_IN_VIDEO)
    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    temp_dir = None
    temp_path = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="miniomni2_server_")
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)

        print(f"Request media: video={use_video} audio={use_audio}")
        answer = run_inference(temp_path, question, temp_dir, use_video, use_audio)
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
    parser = argparse.ArgumentParser(description="Mini-Omni2 Video Analysis Server")
    parser.add_argument("--port", type=int, default=5092, help="Server port (default: 5092)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model()
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
