import torch
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import argparse
from flask import Flask, request, jsonify
import logging
from config.paths import PATHS
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from config.settings import CONFIG
from models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices

app = Flask(__name__)
logger = logging.getLogger("qwen2_5_omni_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    model_log_dir = PATHS.results_logs / "qwen2_5_omni"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_log_dir / "server.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Simplified system prompt
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Global configuration
MODEL_PATH = CONFIG.model("qwen2_5_omni").get("model_path") or "/publicssd/xty/models/Qwen2.5-Omni-7B"
USE_AUDIO_IN_VIDEO = CONFIG.model("qwen2_5_omni").get("use_audio_in_video", True)
MAX_TOKENS = CONFIG.model("qwen2_5_omni").get("max_tokens", 50)
TEMPERATURE = CONFIG.model("qwen2_5_omni").get("temperature", 0.1)

# GPU configuration - supports environment variables and CLI arguments
def get_gpu_id():
    """Resolve the first visible GPU for this process."""
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_id is not None:
        try:
            return int(gpu_id.split(",")[0].strip())
        except ValueError:
            pass
    visible_gpus = configure_cuda_visible_devices(
        CONFIG.model("qwen2_5_omni").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
    )
    if visible_gpus:
        return int(visible_gpus[0])
    return 0

# Global variables
gpu_id = get_gpu_id()
model = None
processor = None
model_loaded = False


def _parse_bool(value, default=True):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def load_model():
    """Load model onto specified GPU"""
    global model, processor, model_loaded, gpu_id

    if model_loaded:
        return

    try:
        print(f"Loading model on GPU {gpu_id}...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=f"cuda:{gpu_id}",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        model_loaded = True
        print(f"Model loaded successfully！Using GPU {gpu_id}")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise


def build_conversation(video_path, question, use_video=True, use_audio=True):
    """Build conversation format"""
    user_content = []
    if use_video:
        user_content.append({"type": "video", "video": video_path})
    elif use_audio:
        user_content.append({"type": "audio", "audio": video_path})
    user_content.append({"type": "text", "text": question})
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def process_video_analysis(video_path, question, use_video, use_audio):
    """Process video analysis"""
    global model, processor
    use_audio_in_video = USE_AUDIO_IN_VIDEO and use_audio
    
    # Build conversation
    conversation = build_conversation(video_path, question, use_video=use_video, use_audio=use_audio)
    
    # Prepare inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    if not use_audio:
        audios = None
    if not use_video:
        images = None
        videos = None
    logger.info("Request media: video=%s audio=%s", use_video, use_audio)
    
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos,
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference with max_tokens limit
    text_ids, _ = model.generate(
        **inputs, 
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True
    )
    result = processor.batch_decode(
        text_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return result[0] if result else ""


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "model_loaded": model_loaded,
        "gpu_id": gpu_id
    })


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze video endpoint - file upload only"""
    global model, processor
    
    if not model_loaded:
        return jsonify({"error": "Model is not loaded"}), 500
    
    temp_dir = None
    temp_path = None
    
    try:
        # File upload only
        if 'video' not in request.files:
            return jsonify({"error": "Video file not uploaded"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        question = request.form.get('question', '')
        use_video = _parse_bool(request.form.get("use_video"), True)
        use_audio = _parse_bool(request.form.get("use_audio"), USE_AUDIO_IN_VIDEO)
        if not question.strip():
            return jsonify({"error": "Question cannot be empty"}), 400
            
        # Save uploaded file to temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)
        
        # Process video analysis
        answer = process_video_analysis(temp_path, question, use_video, use_audio)
        
        # Simplified response format
        return jsonify({
            "status": "success", 
            "answer": answer.strip()
        })
        
    except Exception as e:
        logger.error("Analyze failed: %s", e)
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500
        
    finally:
        # Clean temporary files
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temporary files: {str(e)}")


def parse_args():
    """Parse command-line arguments"""
    default_host = CONFIG.model("qwen2_5_omni").get("host") or "127.0.0.1"
    default_port = CONFIG.model("qwen2_5_omni").get("port") or 5089
    parser = argparse.ArgumentParser(description="Qwen Omni Video Analysis Server")
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=None,
        help="Specify GPU ID (default: 0, or set via CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=default_port,
        help="Server port (default: 5089)"
    )
    parser.add_argument(
        "--host", 
        default=default_host,
        help="Server host address (default: 127.0.0.1)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Update GPU ID if provided via CLI.
    if args.gpu_id is not None:
        gpu_id = args.gpu_id
        # Set environment variable.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU from command-line argument {gpu_id}")
    else:
        print(f"Using GPU {gpu_id} (from env var or default)")
    
    # Load model on startup.
    load_model()
    
    # Start server.
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
