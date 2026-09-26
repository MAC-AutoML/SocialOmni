from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from config.settings import CONFIG
from models.pipeline.types import InferenceRequest, InferenceResult


class SGLangOmniClient:
    """Call an SGLang-Omni OpenAI-compatible ``/v1/chat/completions`` server."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("models.model_server.sglang_omni")

    @property
    def model_name(self) -> str:
        return "sglang_omni"

    @staticmethod
    def _request_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _extract_audio(self, video_path: str | Path) -> str | None:
        """Extract PCM audio for servers configured to consume a separate audio part."""
        try:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output = handle.name
            handle.close()
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-i", str(video_path), "-vn",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            return output
        except (OSError, subprocess.SubprocessError):
            self.logger.warning("Could not extract audio from %s", video_path, exc_info=True)
            return None

    def _build_payload(self, request: InferenceRequest) -> dict[str, Any]:
        config = CONFIG.model("sglang_omni")
        metadata = request.metadata or {}
        model = metadata.get("model") or config.get("model") or os.getenv("SGLANG_OMNI_MODEL")
        if not model:
            raise ValueError("Missing SGLang-Omni model; configure models.sglang_omni.model or SGLANG_OMNI_MODEL.")

        user_prompt = metadata.get("user_prompt") or config.get("user_prompt")
        text = f"{user_prompt}\n\n{request.question}" if user_prompt else request.question
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        use_video = bool(metadata.get("use_video", True))
        use_audio = bool(metadata.get("use_audio", True))
        media: dict[str, list[str]] = {}
        if use_video:
            media["videos"] = [str(request.video_path)]

        audio_path = metadata.get("audio_path")
        temporary_audio: str | None = None
        if use_audio and audio_path:
            media["audios"] = [str(audio_path)]
        elif use_audio and (not use_video or bool(config.get("separate_audio", False))):
            temporary_audio = self._extract_audio(request.video_path)
            if temporary_audio:
                media["audios"] = [temporary_audio]

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "modalities": ["text"],
            "temperature": metadata.get("temperature", config.get("temperature", 0.3)),
            "max_tokens": metadata.get("max_tokens", config.get("max_tokens", 256)),
        }
        payload.update(media)
        if use_video and use_audio and not media.get("audios"):
            payload["use_audio_in_video"] = bool(config.get("use_audio_in_video", True))
        for key in ("video_fps", "video_max_frames", "video_min_pixels", "video_max_pixels", "video_total_pixels"):
            value = metadata.get(key, config.get(key))
            if value is not None:
                payload[key] = value
        if temporary_audio:
            payload["_temporary_audio"] = temporary_audio
        return payload

    @staticmethod
    def _response_text(body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("SGLang-Omni response has no choices")
        content = (choices[0].get("message") or {}).get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content).strip()
        return str(content or "").strip()

    def predict(self, request: InferenceRequest) -> InferenceResult:
        config = CONFIG.model("sglang_omni")
        metadata = request.metadata or {}
        server_url = metadata.get("server_url") or os.getenv("SGLANG_OMNI_SERVER_URL") or config.get("server_url")
        if not server_url:
            raise ValueError("Missing SGLang-Omni server_url; configure models.sglang_omni.server_url or SGLANG_OMNI_SERVER_URL.")
        endpoint = server_url.rstrip("/")
        if not endpoint.endswith("/v1/chat/completions"):
            endpoint += "/v1/chat/completions"

        payload = self._build_payload(request)
        temporary_audio = payload.pop("_temporary_audio", None)
        request_hash = self._request_hash(payload)
        max_retries = int(metadata.get("max_retries", config.get("max_retries", CONFIG.runtime("max_retries", 5))))
        retry_delay = float(metadata.get("retry_delay", config.get("retry_delay", CONFIG.runtime("request_delay", 0.0))))
        timeout = float(metadata.get("timeout", config.get("timeout", 300)))
        last_error = ""
        attempts = 0
        try:
            for attempts in range(1, max(1, max_retries) + 1):
                try:
                    response = requests.post(endpoint, json=payload, timeout=timeout)
                    if response.status_code == 200:
                        answer = self._response_text(response.json())
                        return InferenceResult(
                            answer=answer,
                            raw_response=answer,
                            extra={"request_hash": request_hash, "attempts": attempts, "status_code": 200, "endpoint": endpoint},
                        )
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    if response.status_code < 500 and response.status_code != 429:
                        break
                except (requests.RequestException, ValueError) as exc:
                    last_error = str(exc)
                if attempts < max(1, max_retries):
                    time.sleep(retry_delay)
        finally:
            if temporary_audio:
                try:
                    Path(temporary_audio).unlink()
                except OSError:
                    pass
        raise RuntimeError(f"SGLang-Omni request failed after {attempts} attempt(s): {last_error}")
