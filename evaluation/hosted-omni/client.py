"""Streaming OpenAI-compatible requests with an immutable attempt cache."""

import asyncio
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def video_part(path):
    content = Path(path).read_bytes()
    return {
        "type": "video_url",
        "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(content).decode()
        },
    }, hashlib.sha256(content).hexdigest()


def load_key(path, name="OPENAI_API_KEY"):
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key == name:
            return value.strip().strip("\"'")
    raise ValueError(f"Missing credential field {name}")


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class StreamingClient:
    def __init__(
        self,
        api_key,
        cache_dir,
        base_url=None,
        model="dashscope/qwen3.8-omni-flash",
        max_attempts=3,
        timeout=300,
        trust_env=False,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("Empty API key")
        if not base_url:
            raise ValueError("An API base URL is required")
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.trust_env = trust_env
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "Expected an HTTP(S) endpoint without embedded credentials"
            )
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        self.url = urlunsplit(parsed._replace(path=path, fragment=""))
        self.safe_endpoint = urlunsplit(
            parsed._replace(netloc=parsed.hostname or "", query="", fragment="")
        )

    async def complete(
        self,
        messages,
        media_sha256=None,
        temperature=0.3,
        top_p=1,
        max_tokens=8192,
        enable_thinking=None,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        request_hash = digest({"endpoint": self.url, "payload": payload})
        folder = self.cache_dir / request_hash
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("complete"):
                return result
        manifest = {
            "request_sha256": request_hash,
            "payload_sha256": digest(payload),
            "media_sha256": media_sha256 or [],
            "model": self.model,
            "model_sha256": digest(self.model),
            "endpoint": self.safe_endpoint,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
        }
        atomic_json(folder / "request.json", manifest)
        prior = len(list(folder.glob("attempt-*.json")))
        for retry in range(self.max_attempts):
            number = prior + retry + 1
            record = {
                **manifest,
                "attempt": number,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "events": [],
                "text": "",
                "reasoning": "",
                "finish_reason": None,
                "usage": None,
                "complete": False,
            }
            started = time.monotonic()
            retryable = True
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    trust_env=self.trust_env,
                ) as session:
                    async with session.post(
                        self.url,
                        json=payload,
                        headers={"Authorization": "Bearer " + self.api_key},
                    ) as response:
                        record["http_status"] = response.status
                        if response.status != 200:
                            retryable = response.status in {
                                408,
                                429,
                                500,
                                502,
                                503,
                                504,
                            }
                            raise RuntimeError(f"HTTP {response.status}")
                        done = False
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                continue
                            event = json.loads(data)
                            record["events"].append(event)
                            if event.get("error"):
                                raise RuntimeError("Provider stream error")
                            if event.get("usage"):
                                record["usage"] = event["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("index", 0) != 0:
                                    continue
                                delta = choice.get("delta", {})
                                record["text"] += delta.get("content") or ""
                                record["reasoning"] += (
                                    delta.get("reasoning_content") or ""
                                )
                                if choice.get("finish_reason"):
                                    record["finish_reason"] = choice["finish_reason"]
                        record["stream_done"] = done
                        record["complete"] = done and record["finish_reason"] == "stop"
                        if not record["complete"]:
                            raise RuntimeError("Incomplete stream or non-stop finish")
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                RuntimeError,
                ValueError,
            ) as error:
                record["error_type"] = type(error).__name__
                if isinstance(error, RuntimeError):
                    record["error"] = str(error)
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                # Never serialize credentials, including accidental provider echoes.
                safe_record = json.loads(
                    json.dumps(record).replace(self.api_key, "[REDACTED]")
                )
                atomic_json(folder / f"attempt-{number:04d}.json", safe_record)
            if record["complete"]:
                atomic_json(result_path, safe_record)
                return safe_record
            if not retryable:
                break
            if retry + 1 < self.max_attempts:
                await asyncio.sleep(min(2**retry, 8))
        raise RuntimeError(f"Request failed; inspect {folder}")
