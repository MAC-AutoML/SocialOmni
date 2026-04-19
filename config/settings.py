from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .paths import PATHS


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _default_config() -> Dict[str, Any]:
    return {
        "api": {
            "openai": {"base_url": ""},
        },
        "runtime": {
            "max_retries": 5,
            "request_delay": 0.0,
            "gpu_ids": [],
            "frame_interval_sec": 1,
            "max_frames": 8,
        },
        "models": {
            "defaults": {
                "temperature": 0.3,
                "top_p": 1.0,
                "max_tokens": 256,
                "use_audio_in_video": True,
                "gpu_ids": [],
            }
        },
        "prompts": {"answer_format": "Answer ONLY with the option letter (A, B, C, or D). Do not include any other text."},
        "benchmark": {
            "level1": {
                "model": "",
                "dataset_path": "",
                "video_dir": "",
                "output_dir": "",
                "output_pattern": "results_{model}_level1.json",
                "log_dir": "",
                "modality": "avt",
                "user_prompt": "",
                "max_retries": 5,
                "retry_delay": 3,
                "num_workers": 8,
            }
            ,
            "level2": {
                "model": "",
                "dataset_path": "",
                "video_dir": "",
                "output_dir": "",
                "output_pattern": "results_{model}_level2.json",
                "log_dir": "",
                "modality": "avt",
                "system_prompt": "",
                "user_prompt": "",
                "judge_model": "gpt4o",
                "max_retries": 5,
                "retry_delay": 3,
                "num_workers": 8,
            },
            "level3": {
                "model": "",
                "dataset_path": "",
                "video_dir": "",
                "output_dir": "",
                "output_pattern": "results_{model}_level3.json",
                "log_dir": "",
                "modality": "avt",
                "user_prompt": "",
                "max_retries": 5,
                "retry_delay": 3,
                "num_workers": 8,
            },
        },
    }


class Config:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def api(self, name: str) -> Dict[str, Any]:
        return deepcopy(self._data.get("api", {}).get(name, {}))

    def runtime(self, key: str, default: Any = None) -> Any:
        return self._data.get("runtime", {}).get(key, default)

    def model(self, name: str) -> Dict[str, Any]:
        defaults = deepcopy(self._data.get("models", {}).get("defaults", {}))
        specific = self._data.get("models", {}).get(name, {})
        return _deep_merge(defaults, deepcopy(specific))

    def prompt(self, name: str, default: str = "") -> str:
        return self._data.get("prompts", {}).get(name, default)

    def benchmark(self, path: str, default: Any = None) -> Any:
        return self.get(f"benchmark.{path}", default)


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE")
    if openai_key:
        config.setdefault("api", {}).setdefault("openai", {})["api_key"] = openai_key
    if openai_base:
        config.setdefault("api", {}).setdefault("openai", {})["base_url"] = openai_base

    runtime_frame_interval = os.getenv("SOCIALOMNI_RUNTIME_FRAME_INTERVAL_SEC")
    if runtime_frame_interval:
        try:
            config.setdefault("runtime", {})["frame_interval_sec"] = int(runtime_frame_interval)
        except ValueError:
            pass

    runtime_max_frames = os.getenv("SOCIALOMNI_RUNTIME_MAX_FRAMES")
    if runtime_max_frames is not None and runtime_max_frames != "":
        lowered = runtime_max_frames.strip().lower()
        if lowered in {"none", "null", "-1"}:
            config.setdefault("runtime", {})["max_frames"] = None
        else:
            try:
                config.setdefault("runtime", {})["max_frames"] = int(runtime_max_frames)
            except ValueError:
                pass


def load_config() -> Config:
    load_dotenv(PATHS.config_dir / ".env")
    load_dotenv(PATHS.root / ".env")

    config = _default_config()
    config_path = PATHS.config_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, file_config)

    _apply_env_overrides(config)
    return Config(config)


CONFIG = load_config()
