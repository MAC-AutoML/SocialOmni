from __future__ import annotations

import json
from pathlib import Path

from models.model_server.sglang_omni.client import SGLangOmniClient
from models.pipeline.types import InferenceRequest


def test_build_payload_contains_video_and_audio(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    request = InferenceRequest(
        video_path=str(video),
        question="Choose A.",
        metadata={"model": "test-model", "audio_path": str(audio), "user_prompt": "Answer briefly."},
    )

    payload = SGLangOmniClient()._build_payload(request)
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Answer briefly.\n\nChoose A."}
    assert payload["videos"] == [str(video)]
    assert payload["audios"] == [str(audio)]
    assert payload["modalities"] == ["text"]


def test_predict_parses_openai_response(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    captured: dict = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "A"}}]}

    def fake_post(url: str, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("models.model_server.sglang_omni.client.requests.post", fake_post)
    result = SGLangOmniClient().predict(
        InferenceRequest(video_path=str(video), question="Choose.", metadata={"model": "test-model", "server_url": "http://server:8000"})
    )
    assert result.answer == "A"
    assert len(result.extra["request_hash"]) == 64
    assert captured["url"] == "http://server:8000/v1/chat/completions"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["videos"] == [str(video)]
    json.dumps(captured["json"])
