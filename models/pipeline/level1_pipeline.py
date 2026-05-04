from __future__ import annotations

import json
import re
import logging
import time
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.paths import PATHS
from config.settings import CONFIG
from models.pipeline.model_client import ModelClient
from models.pipeline.modality import add_payload_modality, add_row_modality, modality_metadata, output_path_for
from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.dataset_downloader import ensure_default_dataset_available
from models.utils.openai_compat_tester import GeminiSafetyBlockedError
@dataclass(frozen=True)
class Level1Config:
    dataset_path: Path
    video_dir: Path
    output_path: Path
    log_dir: Path
    max_samples: Optional[int] = None
    start_index: int = 0
    resume: bool = False

class Level1Pipeline:
    """Unified Level1 evaluation flow: build question -> call model -> evaluate -> output results."""

    def __init__(self, omni_test: ModelClient, config: Level1Config) -> None:
        self.omni_test = omni_test
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        model_log_dir = self.config.log_dir / self.omni_test.model_name
        model_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = model_log_dir / f"level1_{self.omni_test.model_name}_{modality_metadata(1)['modality']}_{timestamp}.log"

        logger = logging.getLogger(f"level1_{self.omni_test.model_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_dataset(self) -> List[dict]:
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_request(self, sample: dict) -> InferenceRequest:
        options = sample.get("options") or []
        system_prompt = CONFIG.benchmark("level1.system_prompt", "").strip()
        user_prompt_base = CONFIG.benchmark("level1.user_prompt", "").strip()
        answer_format = CONFIG.prompt("answer_format").strip()
        meta = modality_metadata(1)
        use_video, use_audio = bool(meta["use_video"]), bool(meta["use_audio"])
        asr_content = sample.get("asr_content") or ""
        if not use_audio:
            asr_content = ""

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"[SYSTEM]\n{system_prompt}")
        if asr_content:
            prompt_parts.append(f"[ASR]\n{asr_content}")
        if options:
            prompt_parts.append("Options:\n" + "\n".join(options))
        if user_prompt_base:
            prompt_parts.append(user_prompt_base)
        if answer_format:
            prompt_parts.append(answer_format)

        user_prompt = "\n\n".join(prompt_parts) if prompt_parts else None
        return InferenceRequest(
            video_path=str(self.config.video_dir / sample["video_path"]),
            question=sample["question"],
            options=options,
            metadata={
                "correct_answer": sample.get("correct_answer"),
                "sample_id": sample.get("id"),
                "asr_content": asr_content,
                "user_prompt": user_prompt,
                "use_video": use_video, "use_audio": use_audio,
                "visual_mask": bool(meta.get("visual_mask", False)),
            },
        )

    def _resolve_modality(self) -> tuple[bool, bool]:
        meta = modality_metadata(1)
        return bool(meta["use_video"]), bool(meta["use_audio"])

    def _normalize_answer(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return ""
        if answer[0].upper() in {"A", "B", "C", "D"}:
            return answer[0].upper()
        match = re.search(r"\b([A-D])\b", answer.upper())
        if match:
            return match.group(1)
        return answer

    def _score(self, prediction: str, correct: Optional[str]) -> bool:
        if not correct:
            return False
        return prediction == correct.strip().upper()

    def _is_scored_result(self, row: dict) -> bool:
        return bool(row.get("scored", True))

    def _save_payload(self, payload: dict) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.config.output_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.config.output_path)

    def _is_api_model(self) -> bool:
        return self.omni_test.model_name in {
            "gpt4o",
            "gemini_2_5_flash",
            "gemini_2_5_pro",
            "gemini_3_flash_preview",
            "gemini_3_pro_preview",
        }

    def _resolve_workers(self) -> int:
        # Keep serial execution under current policy to avoid frequent upstream 503
        # errors causing excessive skips under concurrency.
        return 1

    def _resolve_retry_failed_threshold(self) -> int:
        raw = os.getenv("SOCIALOMNI_RETRY_FAILED_THRESHOLD")
        if raw is None:
            raw = CONFIG.runtime("retry_failed_threshold", 1800)
        try:
            return max(1, int(raw))
        except Exception:  # noqa: BLE001
            return 1800

    def _force_retry_failed(self) -> bool:
        raw = os.getenv("SOCIALOMNI_FORCE_RETRY_FAILED")
        if raw is None:
            raw = CONFIG.runtime("force_retry_failed_api", False)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "y", "yes", "on"}

    def _resolve_sample_max_attempts(self) -> int:
        raw = os.getenv("SOCIALOMNI_SAMPLE_MAX_ATTEMPTS")
        if raw is None:
            raw = CONFIG.runtime("sample_max_attempts", None)
        try:
            parsed = int(raw) if raw is not None else 0
        except Exception:  # noqa: BLE001
            parsed = 0
        if self._is_api_model():
            # In API mode, cap attempts per sample by default to prevent a few
            # slow samples from occupying the whole thread pool.
            return parsed if parsed > 0 else 2
        return parsed if parsed > 0 else 0

    def _infer_with_retry(
        self,
        sample: dict,
        base_retry_delay_sec: float,
        max_retry_delay_sec: float,
    ) -> dict:
        request = self._build_request(sample)
        sample_id = sample.get("id")
        attempt = 0
        max_sample_attempts = self._resolve_sample_max_attempts()
        while True:
            attempt += 1
            try:
                result = self.omni_test.predict(request)
                return {"status": "ok", "sample": sample, "result": result}
            except GeminiSafetyBlockedError as exc:
                return {
                    "status": "skip",
                    "sample": sample,
                    "skip_reason": "gemini_safety_block",
                    "skip_error": str(exc)[:1000],
                }
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if max_sample_attempts > 0 and attempt >= max_sample_attempts:
                    return {
                        "status": "skip",
                        "sample": sample,
                        "skip_reason": "retry_exhausted",
                        "skip_error": str(exc)[:1000],
                    }
                delay_sec = min(max_retry_delay_sec, base_retry_delay_sec * (2 ** min(attempt - 1, 6)))
                self.logger.warning(
                    "[%s] inference failed at attempt=%s, retry after %.1fs: %s",
                    sample_id,
                    attempt,
                    delay_sec,
                    exc,
                )
                print(
                    f"[RETRY] model={self.omni_test.model_name} sample_id={sample_id} "
                    f"attempt={attempt} sleep={delay_sec:.1f}s err={str(exc)[:220]}",
                    flush=True,
                )
                time.sleep(delay_sec)

    def run(self) -> dict:
        dataset = self.load_dataset()
        if self.config.start_index > 0:
            dataset = dataset[self.config.start_index :]
        if self.config.max_samples is not None:
            dataset = dataset[: self.config.max_samples]

        results = []
        correct = 0
        total = 0
        skipped_banned = 0
        skipped_failed = 0

        processed_ids = set()
        if self.config.resume and self.config.output_path.exists():
            try:
                with open(self.config.output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                results = existing.get("results", [])
                correct = sum(1 for r in results if self._is_scored_result(r) and r.get("is_correct"))
                total = sum(1 for r in results if self._is_scored_result(r))
                skipped_banned = sum(1 for r in results if r.get("skip_reason") == "gemini_safety_block")
                skipped_failed = sum(1 for r in results if r.get("skip_reason") == "retry_exhausted")

                if self._is_api_model():
                    threshold = self._resolve_retry_failed_threshold()
                    force_retry_failed = self._force_retry_failed()
                    should_retry_failed = force_retry_failed or total < threshold
                    if should_retry_failed:
                        processed_ids = {
                            r.get("id")
                            for r in results
                            if r.get("id") is not None and r.get("skip_reason") != "retry_exhausted"
                        }
                    else:
                        processed_ids = {r.get("id") for r in results if r.get("id") is not None}
                else:
                    processed_ids = {r.get("id") for r in results if r.get("id") is not None}

                self.logger.info(
                    "Resume enabled: processed=%s scored=%s correct=%s skipped_banned=%s skipped_failed=%s",
                    len(results),
                    total,
                    correct,
                    skipped_banned,
                    skipped_failed,
                )
                if self._is_api_model() and skipped_failed > 0:
                    threshold = self._resolve_retry_failed_threshold()
                    force_retry_failed = self._force_retry_failed()
                    should_retry_failed = force_retry_failed or total < threshold
                    self.logger.info(
                        "API resume retry_failed policy: scored=%s threshold=%s force=%s retry_failed=%s",
                        total,
                        threshold,
                        force_retry_failed,
                        should_retry_failed,
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load existing results: %s", exc)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_retry_delay_sec = float(CONFIG.runtime("request_delay", 1.0) or 1.0)
        if base_retry_delay_sec <= 0:
            base_retry_delay_sec = 1.0
        max_retry_delay_sec = 60.0
        result_index_by_id = {r.get("id"): i for i, r in enumerate(results) if r.get("id") is not None}
        pending_samples = [s for s in dataset if not (processed_ids and s.get("id") in processed_ids)]
        remaining_ids = {s.get("id") for s in pending_samples if s.get("id") is not None}

        def _next_pending_id() -> Optional[object]:
            if not remaining_ids:
                return None
            try:
                return min(remaining_ids, key=lambda x: int(x))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                return sorted(remaining_ids, key=lambda x: str(x))[0]

        def _upsert_result(row: dict) -> None:
            sample_id = row.get("id")
            if sample_id is not None and sample_id in result_index_by_id:
                results[result_index_by_id[sample_id]] = row
                return
            if sample_id is not None:
                result_index_by_id[sample_id] = len(results)
            results.append(row)

        # Write initial progress once to avoid reading next_sample_id='-'
        # during early runtime.
        initial_accuracy = (correct / total * 100) if total > 0 else 0.0
        initial_payload = add_payload_modality({
            "model": self.omni_test.model_name,
            "timestamp": run_timestamp,
            "accuracy": initial_accuracy,
            "correct": correct,
            "total": total,
            "processed": len(results),
            "skipped_banned": skipped_banned,
            "skipped_failed": skipped_failed,
            "next_pending_id": _next_pending_id(),
            "results": results,
        }, 1)
        self._save_payload(initial_payload)

        def _consume_outcome(outcome: dict) -> None:
            nonlocal total, correct, skipped_banned, skipped_failed
            sample = outcome["sample"]
            sample_id = sample.get("id")
            if outcome["status"] == "skip":
                if outcome.get("skip_reason") == "gemini_safety_block":
                    skipped_banned += 1
                else:
                    skipped_failed += 1
                _upsert_result(
                    add_row_modality({
                        "id": sample_id,
                        "video_path": sample.get("video_path"),
                        "question": sample.get("question"),
                        "options": sample.get("options"),
                        "correct_answer": sample.get("correct_answer"),
                        "prediction": "",
                        "raw_response": "",
                        "is_correct": False,
                        "scored": False,
                        "skip_reason": outcome.get("skip_reason"),
                        "skip_error": outcome.get("skip_error"),
                    }, 1)
                )
                self.logger.warning(
                    "[%s] skipped due to %s: %s",
                    sample_id,
                    outcome.get("skip_reason"),
                    str(outcome.get("skip_error", ""))[:240],
                )
                print(
                    f"[SKIP] model={self.omni_test.model_name} sample_id={sample_id} reason={outcome.get('skip_reason')}",
                    flush=True,
                )
            else:
                result = outcome["result"]
                prediction = self._normalize_answer(result.answer)
                raw_response = result.raw_response
                is_correct = self._score(prediction, sample.get("correct_answer"))
                total += 1
                if is_correct:
                    correct += 1
                _upsert_result(
                    add_row_modality({
                        "id": sample_id,
                        "video_path": sample.get("video_path"),
                        "question": sample.get("question"),
                        "options": sample.get("options"),
                        "correct_answer": sample.get("correct_answer"),
                        "prediction": prediction,
                        "raw_response": raw_response,
                        "is_correct": is_correct,
                        "scored": True,
                    }, 1)
                )
                self.logger.info(
                    "[%s] prediction=%s correct=%s",
                    sample_id,
                    prediction,
                    sample.get("correct_answer"),
                )

            accuracy = (correct / total * 100) if total > 0 else 0.0
            payload = add_payload_modality({
                "model": self.omni_test.model_name,
                "timestamp": run_timestamp,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "processed": len(results),
                "skipped_banned": skipped_banned,
                "skipped_failed": skipped_failed,
                "next_pending_id": _next_pending_id(),
                "results": results,
            }, 1)
            self._save_payload(payload)
        workers = self._resolve_workers()
        if workers > 1:
            print(
                f"[INFO] API concurrency enabled: model={self.omni_test.model_name}, workers={workers}, pending={len(pending_samples)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                sample_iter = iter(pending_samples)
                pending_futures = {}

                def _submit_next() -> bool:
                    try:
                        next_sample = next(sample_iter)
                    except StopIteration:
                        return False
                    fut = executor.submit(self._infer_with_retry, next_sample, base_retry_delay_sec, max_retry_delay_sec)
                    pending_futures[fut] = next_sample
                    return True

                for _ in range(workers):
                    if not _submit_next():
                        break

                heartbeat_sec = 15.0
                while pending_futures:
                    done, not_done = wait(
                        pending_futures,
                        timeout=heartbeat_sec,
                        return_when=FIRST_COMPLETED,
                    )
                    pending_futures = {fut: pending_futures[fut] for fut in not_done}
                    if not done:
                        queued = len(pending_samples) - len(results) - len(pending_futures)
                        if queued < 0:
                            queued = 0
                        print(
                            f"[PROGRESS] {self.omni_test.model_name}: still running... "
                            f"(processed={len(results)}/{len(dataset)}, inflight={len(pending_futures)}, queued={queued})",
                            flush=True,
                        )
                        continue
                    for future in done:
                        outcome = future.result()
                        _consume_outcome(outcome)
                        sid = outcome["sample"].get("id")
                        if sid in remaining_ids:
                            remaining_ids.discard(sid)
                        _submit_next()
        else:
            for sample in pending_samples:
                outcome = self._infer_with_retry(sample, base_retry_delay_sec, max_retry_delay_sec)
                _consume_outcome(outcome)
                sid = outcome["sample"].get("id")
                if sid in remaining_ids:
                    remaining_ids.discard(sid)

        accuracy = (correct / total * 100) if total > 0 else 0.0
        payload = add_payload_modality({
            "model": self.omni_test.model_name,
            "timestamp": run_timestamp,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "processed": len(results),
            "skipped_banned": skipped_banned,
            "skipped_failed": skipped_failed,
            "next_pending_id": _next_pending_id(),
            "results": results,
        }, 1)
        self._save_payload(payload)
        self.logger.info("Accuracy %.2f%% (%s/%s)", accuracy, correct, total)
        self.logger.info("Processed %s, skipped_banned %s", len(results), skipped_banned)
        self.logger.info("Results saved: %s", self.config.output_path)
        return payload


def default_level1_config(model_name: str) -> Level1Config:
    dataset_path_raw = CONFIG.benchmark("level1.dataset_path", "")
    video_dir_raw = CONFIG.benchmark("level1.video_dir", "")
    log_dir = CONFIG.benchmark("level1.log_dir", "")

    dataset_path = Path(dataset_path_raw) if dataset_path_raw else PATHS.data_level_1 / "dataset.json"
    video_dir = Path(video_dir_raw) if video_dir_raw else PATHS.data_level_1 / "videos"
    output_path = output_path_for(1, model_name)
    log_dir = Path(log_dir) if log_dir else PATHS.results_logs

    if not dataset_path_raw and not video_dir_raw:
        ensure_default_dataset_available("level1", dataset_path, video_dir)

    return Level1Config(
        dataset_path=dataset_path,
        video_dir=video_dir,
        output_path=output_path,
        log_dir=log_dir,
        resume=False,
    )


def run_level1(omni_test: ModelClient) -> dict:
    pipeline = Level1Pipeline(omni_test, default_level1_config(omni_test.model_name))
    return pipeline.run()
