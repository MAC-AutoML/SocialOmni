from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List
from bisect import bisect_left

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG
from models.model_server.local_common.gpu_visibility import configure_cuda_visible_devices

warnings.filterwarnings("ignore")

app = Flask(__name__)

PHYSICAL_GPUS = configure_cuda_visible_devices(
    CONFIG.model("ming").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", [])
)

MODEL_PATH = os.path.expanduser(
    str(CONFIG.model("ming").get("model_path") or os.getenv("MING_MODEL_PATH") or "/publicssd/xty/models/Ming-flash-omni-2.0")
)
BUNDLED_CODE_PATH = str((Path(__file__).resolve().parent / "ming_lib").resolve())
MAX_NEW_TOKENS = int(CONFIG.model("ming").get("max_tokens", 256))

model = None
processor = None
model_loaded = False


def _build_split_device_map() -> Dict[str, int]:
    """Distribute layers across GPUs following original split_model logic to avoid overhead from device_map=auto."""
    world_size = max(1, len(PHYSICAL_GPUS))
    num_layers = 32
    cfg_file = Path(MODEL_PATH) / "config.json"
    if cfg_file.exists():
        try:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            num_layers = int(cfg.get("llm_config", {}).get("num_hidden_layers", num_layers))
        except Exception:  # noqa: BLE001
            pass
    if world_size == 1:
        return {"": 0}

    device_map: Dict[str, int] = {}
    layer_per_gpu = num_layers // world_size
    boundaries = [i * layer_per_gpu for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f"model.model.layers.{i}"] = bisect_left(boundaries, i)
    device_map["vision"] = 0
    device_map["audio"] = 0
    device_map["linear_proj"] = 0
    device_map["linear_proj_audio"] = 0
    device_map["model.model.word_embeddings.weight"] = 0
    device_map["model.model.norm.weight"] = 0
    device_map["model.lm_head.weight"] = 0
    device_map["model.model.norm"] = 0
    device_map[f"model.model.layers.{num_layers - 1}"] = 0
    return device_map


def _sync_runtime_artifacts() -> None:
    """Sync bundled ming_lib runtime dependencies to model directory."""
    bundled_root = Path(BUNDLED_CODE_PATH)
    model_root = Path(MODEL_PATH)
    required_relpaths = [
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    force_overwrite_relpaths = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    # Recursively sync Python files under ming_lib to avoid missing files in dynamic import chains.
    for src in bundled_root.rglob("*.py"):
        relpath = str(src.relative_to(bundled_root)).replace("\\", "/")
        dst = model_root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    for relpath in required_relpaths:
        src = bundled_root / relpath
        dst = model_root / relpath
        if not src.exists():
            continue
        if dst.exists() and relpath not in force_overwrite_relpaths:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


def _load_model() -> None:
    global model, processor, model_loaded
    if model_loaded:
        return

    # Use bundled ming_lib code only.
    if BUNDLED_CODE_PATH not in sys.path:
        sys.path.insert(0, BUNDLED_CODE_PATH)
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")
    print("[ming] attn_implementation=flash_attention_2", flush=True)
    print(f"[ming] HF_ENABLE_PARALLEL_LOADING={os.getenv('HF_ENABLE_PARALLEL_LOADING')}", flush=True)
    print(f"[ming] HF_PARALLEL_LOADING_WORKERS={os.getenv('HF_PARALLEL_LOADING_WORKERS')}", flush=True)

    import torch
    import transformers.integrations.accelerate as tf_accelerate
    import transformers.utils.import_utils as tf_import_utils
    from accelerate import dispatch_model as acc_dispatch_model
    from accelerate import utils as acc_utils
    from accelerate.utils import modeling as acc_utils_modeling
    from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    from processing_bailingmm2 import BailingMM2Processor

    # Compatibility for transformers 5.x requiring accelerate>=1.1.0:
    # Follow official integrations/accelerate.py import paths and patch missing symbols into matching modules.
    # This only affects Ming local service without changing global dependency versions.
    tf_import_utils.is_accelerate_available.cache_clear()

    def _accelerate_runtime_available(_min_version: str = tf_import_utils.ACCELERATE_MIN_VERSION) -> bool:
        ok, _ver = tf_import_utils._is_package_available("accelerate", return_version=True)
        return bool(ok)

    tf_import_utils.is_accelerate_available = _accelerate_runtime_available
    tf_accelerate.is_accelerate_available = _accelerate_runtime_available
    if not hasattr(tf_accelerate, "get_balanced_memory"):
        tf_accelerate.get_balanced_memory = acc_utils.get_balanced_memory
    if not hasattr(tf_accelerate, "get_max_memory"):
        tf_accelerate.get_max_memory = acc_utils.get_max_memory
    if not hasattr(tf_accelerate, "infer_auto_device_map"):
        tf_accelerate.infer_auto_device_map = acc_utils.infer_auto_device_map
    if not hasattr(tf_accelerate, "dispatch_model"):
        tf_accelerate.dispatch_model = acc_dispatch_model
    if not hasattr(tf_accelerate, "clean_device_map"):
        tf_accelerate.clean_device_map = acc_utils_modeling.clean_device_map
    if not hasattr(tf_accelerate, "get_max_layer_size"):
        tf_accelerate.get_max_layer_size = acc_utils_modeling.get_max_layer_size
    if not hasattr(tf_accelerate, "get_module_size_with_ties"):
        # Compatibility implementation when accelerate<1.1 misses this function (following transformers call signature)
        def _fallback_get_module_size_with_ties(tied_params, module_size, module_sizes, modules_to_treat):
            if len(tied_params) < 1:
                return module_size, [], []
            tied_module_names = []
            tied_modules = []

            for tied_param in tied_params:
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if tied_param.startswith(n + ".")][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])

            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

            return module_size_with_ties, tied_module_names, tied_modules

        tf_accelerate.get_module_size_with_ties = _fallback_get_module_size_with_ties

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ming only supports GPU inference.")

    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_build_split_device_map(),
        load_image_gen=False,
    ).to(dtype=torch.bfloat16)

    _sync_runtime_artifacts()

    processor = BailingMM2Processor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    model.eval()
    print(f"[ming] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"[ming] hf_device_map={getattr(model, 'hf_device_map', None)}", flush=True)
    print(f"[ming] input_device={_resolve_input_device()}", flush=True)
    model_loaded = True


def _resolve_input_device() -> str:
    assert model is not None
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for _, dev in device_map.items():
            dev_text = str(dev).lower()
            if "cuda" in dev_text or dev_text.isdigit():
                if dev_text.isdigit():
                    return f"cuda:{dev_text}"
                return dev_text
    try:
        return str(next(model.parameters()).device)
    except Exception:  # noqa: BLE001
        return "cuda:0"


def _generate(messages: List[Dict[str, Any]]) -> str:
    import torch

    assert model is not None and processor is not None

    text = processor.apply_chat_template(messages)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    input_device = _resolve_input_device()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
    ).to(input_device)

    for key in list(inputs.keys()):
        if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
            inputs[key] = inputs[key].to(dtype=torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=False,
            do_sample=False,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return str(output_text).strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": bool(model_loaded)}), 200


@app.route("/analyze", methods=["POST"])
def analyze_video():
    question = request.form.get("question", "").strip()
    use_video = str(request.form.get("use_video", "true")).strip().lower() != "false"

    if not question:
        return jsonify({"error": "Missing question"}), 400

    temp_video_path = None
    try:
        _load_model()

        content: List[Dict[str, str]] = []
        if use_video:
            if "video" not in request.files:
                return jsonify({"error": "Missing video file"}), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                request.files["video"].save(tmp.name)
                temp_video_path = tmp.name
            content.append({"type": "video", "video": temp_video_path})

        content.append({"type": "text", "text": question})
        messages = [{"role": "HUMAN", "content": content}]

        answer = _generate(messages)
        return jsonify({"answer": answer})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=CONFIG.model("ming").get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=CONFIG.model("ming").get("port", 5095))
    args = parser.parse_args()

    _load_model()
    app.run(host=args.host, port=args.port)
