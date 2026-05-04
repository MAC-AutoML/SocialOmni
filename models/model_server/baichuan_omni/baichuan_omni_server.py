import argparse
import json
import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BAICHUAN_ROOT = Path(__file__).resolve().parent / "baichuan_omni_lib"
if str(BAICHUAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BAICHUAN_ROOT))

from config.settings import CONFIG
from models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Set GPU visibility before importing torch
PHYSICAL_GPUS = configure_cuda_visible_devices(
    CONFIG.model("baichuan_omni_1_5").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

os.environ.setdefault("BAICHUAN_OMNI_DISABLE_AUDIO", "1")

MODEL_PATH = CONFIG.model("baichuan_omni_1_5").get("model_path") or "/publicssd/xty/models/Baichuan-Omni-1.5"
MAX_NEW_TOKENS = CONFIG.model("baichuan_omni_1_5").get("max_tokens", 256)
TEMPERATURE = CONFIG.model("baichuan_omni_1_5").get("temperature", 0.3)
TOP_P = 1.0


def _parse_max_frame_num() -> int:
    raw = os.getenv("BAICHUAN_OMNI_MAX_FRAME_NUM")
    if raw is None:
        raw = CONFIG.runtime("max_frames", 8)
    text = str(raw).strip().lower()
    if text in {"", "none", "null"}:
        return 8
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return 8
    return max(1, value)


MAX_FRAME_NUM = _parse_max_frame_num()
MAX_TIME = float(os.getenv("BAICHUAN_OMNI_MAX_TIME", "30"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful and precise multimodal assistant."
ANSWER_RULE = (
    "You must answer ONLY with the option letter (A, B, C, or D). "
    "Do NOT output any other words, symbols, or punctuation."
)

model = None
tokenizer = None
model_loaded = False


def _load_model_classes(model_path: str):
    import importlib

    model_code_path = Path(model_path).resolve()
    if model_code_path.exists():
        if str(model_code_path) not in sys.path:
            sys.path.insert(0, str(model_code_path))
        # Prevent reusing a previously loaded model from baichuan_omni_lib.
        for key in list(sys.modules.keys()):
            if key == "model" or key.startswith("model."):
                del sys.modules[key]
    else:
        if str(BAICHUAN_ROOT) not in sys.path:
            sys.path.insert(0, str(BAICHUAN_ROOT))
        model_code_path = BAICHUAN_ROOT

    print(f"[baichuan_omni] model code path: {model_code_path}", flush=True)
    omni_config = importlib.import_module("model.configuration_omni")
    omni_model = importlib.import_module("model.modeling_omni")
    return omni_config.OmniConfig, omni_model.OmniForCausalLM


def _build_choice_token_ids(text_tokenizer):
    choices = {}
    for letter in ("A", "B", "C", "D"):
        token_ids = set()
        for prefix in ("", " "):
            ids = text_tokenizer.encode(prefix + letter, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
        if not token_ids:
            ids = text_tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
        choices[letter] = sorted(token_ids)
    return choices


def load_model():
    global model, tokenizer, model_loaded

    if model_loaded:
        return

    print("[baichuan_omni] loading: start", flush=True)
    import torch
    import faulthandler
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Baichuan-Omni-1.5 only supports GPU inference.")

    hf_logging.set_verbosity_error()

    print(
        f"[baichuan_omni] cuda: available={torch.cuda.is_available()} count={torch.cuda.device_count()} visible={os.getenv('CUDA_VISIBLE_DEVICES','')}",
        flush=True,
    )

    model_path = os.path.expanduser(MODEL_PATH)
    OmniConfig, OmniForCausalLM = _load_model_classes(model_path)
    print(f"[baichuan_omni] loading: tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("[baichuan_omni] loading: config", flush=True)
    config = OmniConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "video_config"):
        config.video_config.max_frame_num = MAX_FRAME_NUM

    print("[baichuan_omni] loading: weights", flush=True)
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        model = OmniForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[baichuan_omni] loading: device_map failed: {exc}", flush=True)
        model = OmniForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
    faulthandler.cancel_dump_traceback_later()
    if next(model.parameters()).device.type != "cuda":
        print("[baichuan_omni] loading: to cuda", flush=True)
        model = model.cuda()
        torch.cuda.synchronize()

    print("[baichuan_omni] loading: bind processor", flush=True)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model.bind_processor(tokenizer, training=False, relative_path=None)
    print("[baichuan_omni] loading: done", flush=True)
    model_loaded = True


def _build_prompt(question: str, video_path: str | None, use_video: bool) -> str:
    parts = []
    if use_video and video_path:
        video_info = json.dumps({"local": video_path})
        parts.append(f"<video_start_baichuan>{video_info}<video_end_baichuan>")
    if question:
        parts.append(question.strip())
    parts.append(ANSWER_RULE)

    user_text = "\n\n".join([p for p in parts if p])
    return f"<B_SYS>{DEFAULT_SYSTEM_PROMPT}<C_Q>{user_text}<audiotext_start_baichuan><C_A>"


def _move_list_to_device(items, device):
    if items is None:
        return None
    return [item.to(device) for item in items]


def run_inference(video_path: str, question: str, use_video: bool) -> str:
    assert model_loaded and model is not None

    import torch

    prompt = _build_prompt(question, video_path, use_video)
    import time
    t0 = time.time()
    ret = model.processor([prompt])
    t1 = time.time()
    print(f"[baichuan_omni] processor s: {t1 - t0:.2f}", flush=True)

    device = model.main_device
    input_ids = ret.input_ids.to(device)
    attention_mask = ret.attention_mask.to(device) if ret.attention_mask is not None else None
    labels = None
    audios = ret.audios.to(device) if ret.audios is not None else None
    images = _move_list_to_device(ret.images, device)
    videos = _move_list_to_device(ret.videos, device)

    patch_nums = ret.patch_nums.to(device) if ret.patch_nums is not None else None
    images_grid = ret.images_grid
    videos_patch_nums = ret.videos_patch_nums.to(device) if ret.videos_patch_nums is not None else None
    videos_grid = ret.videos_grid
    encoder_length = ret.encoder_length.to(device) if ret.encoder_length is not None else None
    bridge_length = ret.bridge_length.to(device) if ret.bridge_length is not None else None

    if str(os.getenv("BAICHUAN_OMNI_FORCE_CHOICE", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        print("[baichuan_omni] logits start", flush=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                audios=audios,
                images=images,
                patch_nums=patch_nums,
                images_grid=images_grid,
                videos=videos,
                videos_patch_nums=videos_patch_nums,
                videos_grid=videos_grid,
                encoder_length=encoder_length,
                bridge_length=bridge_length,
                use_cache=False,
                return_dict=True,
            )
        end.record()
        torch.cuda.synchronize()
        print(f"[baichuan_omni] logits ms: {start.elapsed_time(end):.2f}", flush=True)
        logits = outputs.logits[0, -1]
        choice_ids = _build_choice_token_ids(tokenizer)
        scores = {k: (logits[ids].max().item() if ids else float("-inf")) for k, ids in choice_ids.items()}
        result = max(scores, key=scores.get)
        del ret, outputs, input_ids, attention_mask, audios, images, videos
    else:
        print("[baichuan_omni] generate start", flush=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                audios=audios,
                images=images,
                patch_nums=patch_nums,
                images_grid=images_grid,
                videos=videos,
                videos_patch_nums=videos_patch_nums,
                videos_grid=videos_grid,
                encoder_length=encoder_length,
                bridge_length=bridge_length,
                max_new_tokens=MAX_NEW_TOKENS,
                max_time=MAX_TIME,
                use_cache=False,
                do_sample=False,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                return_dict_in_generate=True,
            )
        end.record()
        torch.cuda.synchronize()
        print(f"[baichuan_omni] generate ms: {start.elapsed_time(end):.2f}", flush=True)
        print(f"[baichuan_omni] output shape: {tuple(output.sequences.shape)}", flush=True)

        output_ids = output.sequences[0]
        input_len = input_ids.shape[-1]
        if output_ids.shape[-1] > input_len:
            gen_ids = output_ids[input_len:]
        else:
            gen_ids = output_ids

        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        result = raw_text.strip()
        del ret, output, input_ids, attention_mask, audios, images, videos
    torch.cuda.empty_cache()
    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": bool(model_loaded)}), 200


@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "Missing video file"}), 400

    question = request.form.get("question", "")
    use_video = str(request.form.get("use_video", "true")).strip().lower() != "false"

    if not question.strip():
        return jsonify({"error": "Missing question"}), 400

    temp_path = None
    try:
        load_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            request.files["video"].save(tmp.name)
            temp_path = tmp.name

        answer = run_inference(temp_path, question, use_video)
        return jsonify({"answer": answer})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=CONFIG.model("baichuan_omni_1_5").get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=CONFIG.model("baichuan_omni_1_5").get("port", 5094))
    args = parser.parse_args()

    load_model()
    app.run(host=args.host, port=args.port)
