from __future__ import annotations

import os

from config.settings import CONFIG
from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.omni_http_client import OmniHttpClient


class MiniCPMO45Client:
    """Client for the native MiniCPM-o 4.5 AV server."""

    @property
    def model_name(self) -> str:
        return "minicpmo_4_5"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        model_config = CONFIG.model("minicpmo_4_5")
        metadata = request.metadata or {}
        server_url = (
            metadata.get("server_url")
            or os.getenv("MINICPMO_4_5_SERVER_URL")
            or model_config.get("server_url")
        )
        if not server_url:
            raise ValueError(
                "Missing MiniCPM-o 4.5 server_url; configure it in config/config.yaml "
                "or MINICPMO_4_5_SERVER_URL."
            )
        user_prompt = metadata.get("user_prompt") or model_config.get("user_prompt")
        client = OmniHttpClient(server_url)
        raw_answer = client.call_api(
            request.video_path,
            request.question,
            user_prompt=user_prompt,
            use_video=bool(metadata.get("use_video", True)),
            use_audio=bool(metadata.get("use_audio", True)),
            max_retries=CONFIG.runtime("max_retries", 5),
            retry_delay=CONFIG.runtime("request_delay", 0.0),
        )
        return InferenceResult(answer=raw_answer or "", raw_response=raw_answer)
