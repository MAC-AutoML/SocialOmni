from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.paths import PATHS
from config.settings import CONFIG
from models.pipeline.model_client import ModelClient
from models.pipeline.types import InferenceRequest
from models.utils.dataset_downloader import ensure_default_dataset_available
from models.utils.openai_compat_tester import OpenAICompatTester


@dataclass(frozen=True)
class Level2Config:
    dataset_path: Path
    video_dir: Path
    output_path: Path
    log_dir: Path
    max_samples: Optional[int] = None
    start_index: int = 0
    resume: bool = False
    max_retries: int = 5
    retry_delay: float = 3.0


class Level2Pipeline:
    """Unified Level2 evaluation flow (Q1 whether to speak + Q2 what to say)."""

    def __init__(self, omni_test: ModelClient, config: Level2Config) -> None:
        self.omni_test = omni_test
        self.config = config
        self.logger = self._setup_logger()
        self._tmp_dir = Path(tempfile.mkdtemp(prefix=f"vsync_level2_{self.omni_test.model_name}_"))
        self._judge_tester: Optional[OpenAICompatTester] = None
        self._judge_model_cfg: dict[str, Any] = {}

    def _setup_logger(self) -> logging.Logger:
        model_log_dir = self.config.log_dir / self.omni_test.model_name
        model_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = model_log_dir / f"level2_{self.omni_test.model_name}_{timestamp}.log"

        logger = logging.getLogger(f"level2_{self.omni_test.model_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _save_payload(self, payload: dict[str, Any]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.config.output_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.config.output_path)

    def _load_dataset(self) -> list[dict[str, Any]]:
        with self.config.dataset_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            data = payload.get("data", [])
            if isinstance(data, list):
                return data
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported Level2 dataset format: {self.config.dataset_path}")

    def _resolve_modality(self) -> tuple[bool, bool]:
        raw = str(CONFIG.benchmark("level2.modality", "avt")).strip().lower()
        if raw in {"vt", "v", "vision", "vision+text", "video+text"}:
            return True, False
        if raw in {"at", "a", "audio", "audio+text"}:
            return False, True
        return True, True

    def _parse_timestamp_to_seconds(self, timestamp_str: Any) -> float:
        text = str(timestamp_str or "").strip()
        if not text:
            return 0.0
        if text.isdigit():
            return float(text)
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 3:
                try:
                    # Keep consistency with original speak_time_bench protocol:
                    # treat 00:17:00 as 17 seconds.
                    return float(parts[1])
                except Exception:  # noqa: BLE001
                    pass
            if len(parts) == 2:
                try:
                    m = int(parts[0])
                    s = float(parts[1])
                    return m * 60 + s
                except Exception:  # noqa: BLE001
                    pass
        try:
            return float(text)
        except Exception:  # noqa: BLE001
            return 0.0

    def _cut_video_at_timestamp(self, input_video: Path, timestamp_sec: float, output_video: Path) -> bool:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i",
            str(input_video),
            "-t",
            str(max(0.0, timestamp_sec)),
            "-c",
            "copy",
            "-y",
            str(output_video),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            return result.returncode == 0 and output_video.exists() and output_video.stat().st_size > 0
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("ffmpeg cut failed: %s", exc)
            return False

    def _normalize_q1_prediction(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        up = s.upper()
        if up[0] in {"A", "B"}:
            return up[0]
        m = re.search(r"\b([AB])\b", up)
        if m:
            return m.group(1)
        low = s.lower()
        if "yes" in low:
            return "A"
        if "no" in low:
            return "B"
        return ""

    def _build_q1_prompt(self, sample: dict[str, Any]) -> str:
        q1 = sample.get("question_1", {})
        option_a = (q1.get("option_A") or "YES").strip()
        option_b = (q1.get("option_B") or "NO").strip()
        asr_content = (sample.get("full_asr") or "").strip()
        system_prompt = (CONFIG.benchmark("level2.system_prompt", "") or "").strip()
        user_prompt_base = (CONFIG.benchmark("level2.user_prompt", "") or "").strip()
        answer_format = (CONFIG.prompt("answer_format", "") or "").strip()

        parts: list[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}")
        if asr_content:
            parts.append(f"[ASR]\n{asr_content}")
        parts.append(f"Options:\nA. {option_a}\nB. {option_b}")
        if user_prompt_base:
            parts.append(user_prompt_base)
        if answer_format:
            parts.append(answer_format)
        if not answer_format:
            parts.append("Answer ONLY with the option letter (A or B).")
        return "\n\n".join(parts)

    def _build_q2_prompt(self, sample: dict[str, Any]) -> str:
        asr_content = (sample.get("full_asr") or "").strip()
        system_prompt = (CONFIG.benchmark("level2.system_prompt", "") or "").strip()
        user_prompt_base = (CONFIG.benchmark("level2.user_prompt", "") or "").strip()

        parts: list[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}")
        if asr_content:
            parts.append(f"[ASR]\n{asr_content}")
        if user_prompt_base:
            parts.append(user_prompt_base)
        parts.append("Provide a concise natural utterance only, without explanations.")
        return "\n\n".join(parts)

    def _get_judge(self) -> tuple[OpenAICompatTester, dict[str, Any], str]:
        if self._judge_tester is not None:
            judge_model_name = self._judge_model_cfg.get("model_name", "gpt-4o")
            return self._judge_tester, self._judge_model_cfg, judge_model_name

        judge_model_key = str(CONFIG.benchmark("level2.judge_model", "gpt4o")).strip() or "gpt4o"
        judge_model_cfg = CONFIG.model(judge_model_key)
        judge_model_name = judge_model_cfg.get("model_name", "gpt-4o")
        api_cfg = CONFIG.api("openai")
        self._judge_tester = OpenAICompatTester(
            model_name=judge_model_name,
            api_base=api_cfg.get("base_url"),
            api_key=api_cfg.get("api_key"),
        )
        self._judge_model_cfg = judge_model_cfg
        return self._judge_tester, judge_model_cfg, judge_model_name

    def _parse_judge_score(self, text: str) -> int:
        m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
        if not m:
            return 0
        value = max(0.0, min(100.0, float(m.group(0))))
        buckets = [0, 25, 50, 75, 100]
        return min(buckets, key=lambda b: abs(b - value))

    def _judge_q2(self, reference_answer: str, candidate_answer: str, video_path_for_api: str) -> int:
        if not candidate_answer.strip():
            return 0
        tester, model_cfg, _ = self._get_judge()
        judge_prompt = (
            "You are a strict evaluator for dialog continuation.\n"
            "Score candidate answer against reference on scale {0, 25, 50, 75, 100}.\n"
            "Criteria: semantic match, intent correctness, key info completeness.\n"
            "Output ONLY one number.\n\n"
            f"[Reference]\n{reference_answer}\n\n"
            f"[Candidate]\n{candidate_answer}\n"
        )
        raw = tester.call(
            video_path=video_path_for_api,
            question="Score the candidate answer.",
            user_prompt=judge_prompt,
            model_params=model_cfg,
            include_images=False,
        )
        return self._parse_judge_score(raw)

    def _infer_with_retry(self, request: InferenceRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.config.max_retries) + 1):
            try:
                result = self.omni_test.predict(request)
                answer = (result.answer or "").strip()
                if not answer:
                    raise RuntimeError("empty model answer")
                return answer
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max(1, self.config.max_retries):
                    break
                sleep_s = max(0.1, float(self.config.retry_delay)) * (2 ** (attempt - 1))
                self.logger.warning("infer failed (attempt=%s): %s", attempt, exc)
                time.sleep(sleep_s)
        assert last_error is not None
        raise last_error

    def _is_effective_row(self, row: dict[str, Any]) -> bool:
        if not isinstance(row, dict):
            return False
        if row.get("error"):
            return False
        q1_pred = str(row.get("q1_prediction", "") or "").strip().upper()
        q1_resp = str(row.get("q1_response", "") or "").strip()
        if q1_pred not in {"A", "B"}:
            return False
        if not q1_resp:
            return False
        return True

    def _evaluate_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        video_id = str(sample.get("video_id", "")).strip()
        video_file = str(sample.get("video_file", "")).strip()
        if not video_file:
            raise ValueError(f"Missing video_file for sample {video_id}")

        original_video_path = self.config.video_dir / video_file
        if not original_video_path.exists():
            raise FileNotFoundError(f"Video not found: {original_video_path}")

        q1 = sample.get("question_1", {})
        q2 = sample.get("question_2", {})
        timestamp = self._parse_timestamp_to_seconds(q1.get("timestamp"))
        cut_video_path = self._tmp_dir / f"{video_id}_cut.mp4"
        if not self._cut_video_at_timestamp(original_video_path, timestamp, cut_video_path):
            raise RuntimeError(f"Failed to cut video for sample {video_id} at {timestamp}s")

        use_video, use_audio = self._resolve_modality()

        q1_request = InferenceRequest(
            video_path=str(cut_video_path),
            question=str(q1.get("question", "")),
            options=[f"A. {q1.get('option_A', 'YES')}", f"B. {q1.get('option_B', 'NO')}"],
            metadata={
                "sample_id": video_id,
                "user_prompt": self._build_q1_prompt(sample),
                "use_video": use_video,
                "use_audio": use_audio,
            },
        )
        q1_raw = self._infer_with_retry(q1_request)
        q1_pred = self._normalize_q1_prediction(q1_raw)
        q1_answer = str(q1.get("correct_answer", "")).strip().upper()
        q1_correct = bool(q1_pred and q1_answer and q1_pred == q1_answer)

        q2_question = str(q2.get("question", "")).strip()
        q2_reference = str(q2.get("answer", "") or "").strip()
        q2_response = ""
        q2_score: Optional[int] = None

        if q1_answer == "A":
            if q1_correct:
                q2_request = InferenceRequest(
                    video_path=str(cut_video_path),
                    question=q2_question,
                    metadata={
                        "sample_id": video_id,
                        "user_prompt": self._build_q2_prompt(sample),
                        "use_video": use_video,
                        "use_audio": use_audio,
                    },
                )
                q2_response = self._infer_with_retry(q2_request)
                q2_score = self._judge_q2(q2_reference, q2_response, str(original_video_path))
            else:
                q2_score = 0
        else:
            if q1_correct:
                q2_score = None
            else:
                q2_score = 0

        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "q1_correct": q1_correct,
            "q1_prediction": q1_pred,
            "q1_answer": q1_answer,
            "q1_response": q1_raw,
            "q2_score": q2_score,
            "q2_response": q2_response,
            "q2_reference": q2_reference,
        }

    def _build_payload(self, run_timestamp: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        q1_total = len(results)
        q1_correct = sum(1 for r in results if r.get("q1_correct"))
        q1_accuracy = (q1_correct / q1_total * 100.0) if q1_total > 0 else 0.0
        q2_scores = [
            float(r["q2_score"])
            for r in results
            if isinstance(r.get("q2_score"), (int, float)) and float(r.get("q2_score")) > 0
        ]
        q2_avg = (sum(q2_scores) / len(q2_scores)) if q2_scores else 0.0
        return {
            "model": self.omni_test.model_name,
            "timestamp": run_timestamp,
            "q1_accuracy": q1_accuracy,
            "q1_correct": q1_correct,
            "q1_total": q1_total,
            "q2_avg_score": q2_avg,
            "q2_count": len(q2_scores),
            "results": results,
        }

    def run(self) -> dict[str, Any]:
        dataset = self._load_dataset()
        if self.config.start_index > 0:
            dataset = dataset[self.config.start_index :]
        if self.config.max_samples is not None:
            dataset = dataset[: self.config.max_samples]

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results: list[dict[str, Any]] = []
        completed_video_ids: set[str] = set()

        if self.config.resume and self.config.output_path.exists():
            try:
                with self.config.output_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                results = existing.get("results", []) if isinstance(existing, dict) else []
                completed_video_ids = {
                    str(r.get("video_id", "")).strip()
                    for r in results
                    if r.get("video_id") and self._is_effective_row(r)
                }
                self.logger.info("Resume enabled: already completed=%s", len(completed_video_ids))
                print(
                    f"[PROGRESS] {self.omni_test.model_name}: resume enabled, "
                    f"completed={len(completed_video_ids)}"
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load existing results: %s", exc)

        total_samples = len(dataset)
        for idx, sample in enumerate(dataset, start=1):
            video_id = str(sample.get("video_id", "")).strip()
            if video_id and video_id in completed_video_ids:
                continue

            try:
                row = self._evaluate_one(sample)
                results.append(row)
                if video_id:
                    completed_video_ids.add(video_id)
                self.logger.info(
                    "[%s/%s] %s q1_correct=%s q2_score=%s",
                    idx,
                    total_samples,
                    video_id,
                    row.get("q1_correct"),
                    row.get("q2_score"),
                )
                print(
                    f"[PROGRESS] {self.omni_test.model_name}: "
                    f"[{idx}/{total_samples}] video_id={video_id} "
                    f"q1_correct={row.get('q1_correct')} q2_score={row.get('q2_score')}"
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error("[%s/%s] %s failed: %s", idx, total_samples, video_id, exc)
                results.append(
                    {
                        "video_id": video_id,
                        "timestamp": self._parse_timestamp_to_seconds(sample.get("question_1", {}).get("timestamp")),
                        "q1_correct": False,
                        "q1_prediction": "",
                        "q1_answer": str(sample.get("question_1", {}).get("correct_answer", "")).strip().upper(),
                        "q1_response": "",
                        "q2_score": 0,
                        "q2_response": "",
                        "q2_reference": str(sample.get("question_2", {}).get("answer", "") or "").strip(),
                        "error": str(exc)[:1000],
                    }
                )
                print(
                    f"[PROGRESS] {self.omni_test.model_name}: "
                    f"[{idx}/{total_samples}] video_id={video_id} failed={str(exc)[:160]}"
                )

            payload = self._build_payload(run_timestamp, results)
            self._save_payload(payload)

        payload = self._build_payload(run_timestamp, results)
        self._save_payload(payload)
        self.logger.info(
            "Level2 done: q1_accuracy=%.2f%% (%s/%s), q2_avg=%.2f, q2_count=%s, output=%s",
            payload["q1_accuracy"],
            payload["q1_correct"],
            payload["q1_total"],
            payload["q2_avg_score"],
            payload["q2_count"],
            self.config.output_path,
        )
        return payload


def default_level2_config(model_name: str) -> Level2Config:
    dataset_path_raw = CONFIG.benchmark("level2.dataset_path", "")
    video_dir_raw = CONFIG.benchmark("level2.video_dir", "")
    output_dir = CONFIG.benchmark("level2.output_dir", "")
    output_pattern = CONFIG.benchmark("level2.output_pattern", "results_{model}_level2.json")
    log_dir = CONFIG.benchmark("level2.log_dir", "")
    max_retries = int(CONFIG.benchmark("level2.max_retries", 5) or 5)
    retry_delay = float(CONFIG.benchmark("level2.retry_delay", 3) or 3)

    dataset = Path(dataset_path_raw) if dataset_path_raw else PATHS.data_dir / "level_2" / "annotations.json"
    videos = Path(video_dir_raw) if video_dir_raw else PATHS.data_dir / "level_2" / "videos"
    output_base = Path(output_dir) if output_dir else PATHS.results_dir
    output_path = output_base / output_pattern.format(model=model_name)
    logs = Path(log_dir) if log_dir else PATHS.results_logs

    if not dataset_path_raw and not video_dir_raw:
        ensure_default_dataset_available("level2", dataset, videos)

    return Level2Config(
        dataset_path=dataset,
        video_dir=videos,
        output_path=output_path,
        log_dir=logs,
        resume=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def run_level2(omni_test: ModelClient) -> dict[str, Any]:
    pipeline = Level2Pipeline(omni_test, default_level2_config(omni_test.model_name))
    return pipeline.run()
