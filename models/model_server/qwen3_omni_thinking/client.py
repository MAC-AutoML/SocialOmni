from __future__ import annotations

import os

from config.settings import CONFIG
from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.omni_http_client import OmniHttpClient


class Qwen3OmniThinkingClient:
    @property
    def model_name(self) -> str:
        return "qwen3_omni_thinking"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        model_config = CONFIG.model("qwen3_omni_thinking")
        server_url = request.metadata.get("server_url") if request.metadata else None
        server_url = server_url or model_config.get("server_url") or os.getenv("QWEN3_OMNI_THINKING_SERVER_URL")
        if not server_url:
            raise ValueError("Missing Qwen3-Omni Thinking server_url. Please configure it in config/config.yaml or environment variables.")

        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        user_prompt = user_prompt or model_config.get("user_prompt")
        use_video = True
        use_audio = True
        if request.metadata:
            use_video = bool(request.metadata.get("use_video", True))
            use_audio = bool(request.metadata.get("use_audio", True))
        visual_mask = bool(request.metadata.get("visual_mask", False)) if request.metadata else False

        client = OmniHttpClient(server_url)
        raw_answer = client.call_api(
            request.video_path,
            request.question,
            user_prompt=user_prompt,
            use_video=use_video,
            use_audio=use_audio,
            visual_mask=visual_mask,
            max_retries=CONFIG.runtime("max_retries", 5),
            retry_delay=CONFIG.runtime("request_delay", 0.0),
        )
        return InferenceResult(answer=raw_answer or "", raw_response=raw_answer)
