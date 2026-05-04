#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CONFIG
from models.pipeline.modality import modality_metadata, output_path_for


LOCAL_MODEL_SERVER_SCRIPT = {
    "qwen3_omni": "models/model_server/qwen3_omni/qwen3_omni_server.py",
    "qwen3_omni_thinking": "models/model_server/qwen3_omni_thinking/qwen3_omni_thinking_server.py",
    "qwen2_5_omni": "models/model_server/qwen2_5_omni/qwen_omni_server.py",
    "miniomni_2": "models/model_server/miniomni_2/miniomni2_server.py",
    "omnivinci": "models/model_server/omnivinci/omnivinci_server.py",
    "vita_1_5": "models/model_server/vita/vita_server.py",
    "baichuan_omni_1_5": "models/model_server/baichuan_omni/baichuan_omni_server.py",
    "ming": "models/model_server/ming/ming_server.py",
}


@dataclass
class ServerEndpoint:
    host: str
    port: int
    base_url: str


def is_local_model(model_name: str) -> bool:
    return model_name in LOCAL_MODEL_SERVER_SCRIPT


def resolve_endpoint(model_name: str) -> ServerEndpoint:
    model_cfg = CONFIG.model(model_name)
    server_url = str(model_cfg.get("server_url", "")).strip()
    if server_url:
        parsed = urlparse(server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or int(model_cfg.get("port", 0) or 0)
        if port <= 0:
            raise ValueError(f"{model_name} server_url does not include a valid port: {server_url}")
        return ServerEndpoint(host=host, port=port, base_url=server_url.rstrip("/"))

    host = str(model_cfg.get("host", "127.0.0.1")).strip()
    port = int(model_cfg.get("port", 0) or 0)
    if port <= 0:
        raise ValueError(f"{model_name} is missing a valid port configuration")
    return ServerEndpoint(host=host, port=port, base_url=f"http://{host}:{port}")


def server_health_ok(base_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout_sec)
        if resp.status_code != 200:
            return False
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if isinstance(payload, dict) and "model_loaded" in payload:
            return bool(payload.get("model_loaded"))
        return True
    except Exception:  # noqa: BLE001
        return False


def start_server(model_name: str, endpoint: ServerEndpoint, log_path: Path) -> subprocess.Popen:
    script_rel = LOCAL_MODEL_SERVER_SCRIPT[model_name]
    script_abs = ROOT / script_rel
    if not script_abs.exists():
        raise FileNotFoundError(f"Service script not found: {script_abs}")

    cmd = [
        sys.executable,
        str(script_abs),
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
    ]
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc


def wait_server_ready(
    base_url: str,
    timeout_sec: int,
    interval_sec: float,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            hint = f" (see {log_path})" if log_path else ""
            raise RuntimeError(f"Server process exited before readiness check passed{hint}")
        if server_health_ok(base_url):
            return True
        time.sleep(interval_sec)
    return False


def _safe_read_progress(result_file: Path) -> tuple[int, int, int, int, int, str]:
    try:
        if not result_file.exists():
            return 0, 0, 0, 0, 0, "-"
        data = json.loads(result_file.read_text(encoding="utf-8"))
        processed = int(data.get("processed", 0) or 0)
        total = int(data.get("total", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        skipped_banned = int(data.get("skipped_banned", 0) or 0)
        skipped_failed = int(data.get("skipped_failed", 0) or 0)
        next_pending_id = data.get("next_pending_id")
        next_pending_id_text = "-" if next_pending_id is None else str(next_pending_id)
        return processed, total, correct, skipped_banned, skipped_failed, next_pending_id_text
    except Exception:  # noqa: BLE001
        return 0, 0, 0, 0, 0, "-"


def run_benchmark(
    model_name: str,
    max_samples: Optional[int],
    timeout_sec: int,
    output_path: Path,
    resume: bool,
    progress_interval: int,
    level: int,
) -> tuple[int, str]:
    runner = "run_benchmark.py" if level == 1 else "run_benchmark_level2.py"
    cmd = [
        sys.executable,
        runner,
        "--model",
        model_name,
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if level == 2 and resume:
        cmd.append("--resume")
    env = os.environ.copy()
    env["SOCIALOMNI_ROOT"] = str(ROOT)

    # Temporarily overriding output directory is not supported via environment variables.
    # Keep runner behavior unchanged and use standard files under results/.
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    if level == 1 and proc.stdin is not None:
        proc.stdin.write("y\n" if resume else "n\n")
        proc.stdin.flush()
        proc.stdin.close()

    output_lines: list[str] = []

    def _consume_stdout() -> None:
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            print(line, end="")

    t = threading.Thread(target=_consume_stdout, daemon=True)
    t.start()

    result_file = output_path_for(level, model_name)
    next_tick = time.time() + max(5, progress_interval)
    last_processed = -1

    start = time.time()
    try:
        while proc.poll() is None:
            now = time.time()
            if now - start > timeout_sec:
                proc.kill()
                raise TimeoutError(f"benchmark timeout ({timeout_sec}s): {' '.join(cmd)}")

            if now >= next_tick:
                if level == 1:
                    processed, total, correct, skipped_banned, skipped_failed, next_pending_id = _safe_read_progress(result_file)
                    if processed != last_processed:
                        print(
                            f"[PROGRESS] {model_name}: processed={processed}/{max_samples or 'all'} "
                            f"(scored={total}, correct={correct}, skipped_banned={skipped_banned}, "
                            f"skipped_failed={skipped_failed}, next_sample_id={next_pending_id})"
                        )
                        last_processed = processed
                    else:
                        print(
                            f"[PROGRESS] {model_name}: still running... "
                            f"(processed={processed}/{max_samples or 'all'}, scored={total}, next_sample_id={next_pending_id})"
                        )
                else:
                    try:
                        data = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
                        rows = data.get("results", []) if isinstance(data, dict) else []
                        processed = len(rows) if isinstance(rows, list) else 0
                        q1_total = int(data.get("q1_total", processed) or processed) if isinstance(data, dict) else processed
                        q1_correct = int(data.get("q1_correct", 0) or 0) if isinstance(data, dict) else 0
                        q2_count = int(data.get("q2_count", 0) or 0) if isinstance(data, dict) else 0
                    except Exception:  # noqa: BLE001
                        processed, q1_total, q1_correct, q2_count = 0, 0, 0, 0
                    if processed != last_processed:
                        print(
                            f"[PROGRESS] {model_name}: processed={processed}/{max_samples or 'all'} "
                            f"(q1_correct={q1_correct}, q1_total={q1_total}, q2_count={q2_count})"
                        )
                        last_processed = processed
                    else:
                        print(
                            f"[PROGRESS] {model_name}: still running... "
                            f"(processed={processed}/{max_samples or 'all'}, q1_correct={q1_correct}, q2_count={q2_count})"
                        )
                next_tick = now + max(5, progress_interval)
            time.sleep(1.0)
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        raise

    t.join(timeout=3)
    output = "".join(output_lines)
    return int(proc.returncode or 0), output


def read_result_status(model_name: str, level: int) -> tuple[int, int, float, str, bool]:
    result_file = output_path_for(level, model_name)
    if not result_file.exists():
        return 0, 0, 0.0, "", False
    data = json.loads(result_file.read_text(encoding="utf-8"))
    if level == 1:
        total = int(data.get("total", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        accuracy = float(data.get("accuracy", 0.0) or 0.0)
        # Find a sample with valid output first to avoid false FAIL_INFER from empty leading rows.
        probe = {}
        for row in data.get("results") or []:
            if not bool(row.get("scored", True)):
                continue
            pred = str(row.get("prediction", "") or "").strip()
            raw = row.get("raw_response")
            if pred and raw:
                probe = row
                break
            if not probe:
                probe = row

        prediction = str(probe.get("prediction", "") or "")
        raw_response = probe.get("raw_response")
        raw_present = bool(raw_response)
        return total, correct, accuracy, prediction, raw_present

    total = int(data.get("q1_total", 0) or 0)
    correct = int(data.get("q1_correct", 0) or 0)
    accuracy = float(data.get("q1_accuracy", 0.0) or 0.0)
    rows = data.get("results", []) if isinstance(data, dict) else []
    probe = {}
    for row in rows or []:
        pred = str(row.get("q1_prediction", "") or "").strip()
        raw = str(row.get("q1_response", "") or "").strip()
        if pred and raw:
            probe = row
            break
        if not probe:
            probe = row
    prediction = str(probe.get("q1_prediction", "") or "")
    raw_present = bool(str(probe.get("q1_response", "") or "").strip())
    return total, correct, accuracy, prediction, raw_present


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local model server and benchmark in one Slurm-friendly process")
    parser.add_argument("--model", required=True, help="Model name, e.g. qwen3_omni / gpt4o")
    parser.add_argument("--max-samples", type=int, default=None, help="Number of samples. Omit to run all samples.")
    parser.add_argument("--server-timeout", type=int, default=900, help="Server readiness timeout in seconds, default 900")
    parser.add_argument("--test-timeout", type=int, default=1200, help="Test execution timeout in seconds, default 1200")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Health-check polling interval, default 2s")
    parser.add_argument("--reuse-server", action="store_true", help="Reuse an already running local service instead of starting one")
    parser.add_argument("--resume", action="store_true", help="Auto-resume when existing output is detected")
    parser.add_argument("--progress-interval", type=int, default=15, help="Progress print interval in seconds, default 15")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1, help="Evaluation level: 1 or 2, default 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model.strip()
    modality = modality_metadata(args.level)["modality"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "smoke_tests" / f"{model_name}_{modality}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"
    test_log = out_dir / "test.log"

    started_by_script = False
    server_proc: Optional[subprocess.Popen] = None

    status = "FAIL"
    exit_code = 1
    message = ""

    try:
        if is_local_model(model_name):
            endpoint = resolve_endpoint(model_name)
            print(f"[INFO] Local model: {model_name} -> {endpoint.base_url}")

            if server_health_ok(endpoint.base_url):
                if not args.reuse_server:
                    raise RuntimeError(
                        f"Detected an existing service at {endpoint.base_url}. "
                        "Stop it first or pass --reuse-server explicitly."
                    )
                print("[INFO] Detected running server, reusing existing service.")
            else:
                print("[INFO] Starting local service inside this job...")
                server_proc = start_server(model_name, endpoint, server_log)
                started_by_script = True
                ready = wait_server_ready(
                    base_url=endpoint.base_url,
                    timeout_sec=args.server_timeout,
                    interval_sec=args.poll_interval,
                    proc=server_proc,
                    log_path=server_log,
                )
                if not ready:
                    raise RuntimeError(f"Server startup timeout: {endpoint.base_url} (see {server_log})")
                print("[INFO] Service is ready, starting test.")
        else:
            print(f"[INFO] API model: {model_name}, no local server required.")

        cmd_code, output = run_benchmark(
            model_name=model_name,
            max_samples=args.max_samples,
            timeout_sec=args.test_timeout,
            output_path=out_dir,
            resume=args.resume,
            progress_interval=args.progress_interval,
            level=args.level,
        )
        test_log.write_text(output, encoding="utf-8")

        if cmd_code != 0:
            raise RuntimeError(f"Test command failed, exit={cmd_code} (see {test_log})")

        total, correct, accuracy, prediction, raw_present = read_result_status(model_name, level=args.level)
        if total < 1:
            status = "FAIL_EMPTY"
            exit_code = 2
            message = "No samples found in result file"
        elif prediction and raw_present:
            status = "PASS"
            exit_code = 0
            message = f"prediction={prediction}, accuracy={accuracy:.2f}%"
        else:
            status = "FAIL_INFER"
            exit_code = 3
            message = "Command succeeded but no valid prediction/raw response found"

        print(f"[{status}] {model_name}: {message}")

    except Exception as exc:  # noqa: BLE001
        status = "FAIL"
        exit_code = 1
        message = str(exc)
        (out_dir / "error.log").write_text(message, encoding="utf-8")
        print(f"[FAIL] {model_name}: {message}")

    finally:
        if started_by_script and server_proc:
            stop_server(server_proc)
            print("[INFO] Stopped local server started by this script.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
