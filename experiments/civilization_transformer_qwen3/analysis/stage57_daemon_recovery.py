from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any

from .stage55_service_deployment import Stage55ServiceClient
from .stage56_daemon_operations import (
    Stage56DaemonConfig,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE56_OUTPUT_DIR,
    port_available,
    read_pid_status,
    rotate_log_file,
    write_stage56_daemon_bundle,
    validate_stage56_daemon_bundle,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage57_daemon_recovery")


@dataclass(frozen=True)
class Stage57RecoveryConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    preferred_device: str = "cuda"
    max_length: int = 384
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B"
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json"
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    )
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    daemon_output_dir: str = str(DEFAULT_STAGE56_OUTPUT_DIR)
    log_dir: str = "experiments/civilization_transformer_qwen3/artifacts/logs"
    pid_file: str = "experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.pid"
    log_file: str = "experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.log"
    readiness_timeout_seconds: float = 240.0
    max_log_bytes: int = 1024
    log_backup_count: int = 3
    python_bin: str = ".venv/bin/python"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_script(script: str | Path, *, cwd: str | Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(script)],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "script": str(script),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _wait_http_ready(host: str, port: int, *, timeout_seconds: float) -> dict[str, Any]:
    client = Stage55ServiceClient(f"http://{host}:{port}", timeout=10.0)
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            health = client.health()
            if health.get("ready") is True:
                return {"ready": True, "health": health}
        except Exception as error:  # local readiness probe
            last_error = str(error)
        time.sleep(0.5)
    return {"ready": False, "last_error": last_error}


def _example_payload() -> dict[str, Any]:
    return {
        "id": "stage57-recovery",
        "text": "Evaluate recovery after daemon restart.",
        "memory_items": ["memory supports approval"],
        "rule_items": ["rule permits approval"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "adapter_disabled"],
    }


def _predict_ok(host: str, port: int) -> dict[str, Any]:
    client = Stage55ServiceClient(f"http://{host}:{port}", timeout=60.0)
    payload = client.predict(_example_payload())
    ok = payload.get("status") == "ok" and all(row.get("status") == "ok" for row in payload.get("rows", []))
    return {"ok": ok, "payload": payload}


def _kill_process(pid: int, *, sig: int = signal.SIGKILL) -> dict[str, Any]:
    try:
        os.kill(pid, sig)
    except OSError as error:
        return {"status": "error", "pid": pid, "error_type": type(error).__name__, "error": str(error)}
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return {"status": "ok", "pid": pid, "signal": sig}
        time.sleep(0.2)
    return {"status": "ok", "pid": pid, "signal": sig}


def _make_daemon_config(config: Stage57RecoveryConfig) -> Stage56DaemonConfig:
    return Stage56DaemonConfig(
        host=config.host,
        port=config.port,
        preferred_device=config.preferred_device,
        max_length=config.max_length,
        package_manifest=config.package_manifest,
        centroid_bundle=config.centroid_bundle,
        model_path=config.model_path,
        output_dir=config.daemon_output_dir,
        log_dir=config.log_dir,
        pid_file=config.pid_file,
        log_file=config.log_file,
        python_bin=config.python_bin,
        readiness_timeout_seconds=config.readiness_timeout_seconds,
        max_log_bytes=config.max_log_bytes,
        log_backup_count=config.log_backup_count,
    )


def run_stage57_recovery_smoke(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage57RecoveryConfig = Stage57RecoveryConfig(),
    real: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daemon_config = _make_daemon_config(config)
    manifest = write_stage56_daemon_bundle(output_dir=config.daemon_output_dir, config=daemon_config)
    validation = validate_stage56_daemon_bundle(output_dir=config.daemon_output_dir)
    files = manifest["files"]
    start_script = files["start_script"]
    stop_script = files["stop_script"]
    log_file = Path(config.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("x" * (config.max_log_bytes + 16), encoding="utf-8")
    rotation_before_start = rotate_log_file(log_file, max_bytes=config.max_log_bytes, backup_count=config.log_backup_count)

    # Stop any stale service first. This is idempotent by design.
    pre_stop = _run_script(stop_script, cwd=".", timeout=60.0)
    start = _run_script(start_script, cwd=".", timeout=config.readiness_timeout_seconds + 30.0)
    ready_after_start = _wait_http_ready(config.host, config.port, timeout_seconds=config.readiness_timeout_seconds)
    pid_status_after_start = read_pid_status(config.pid_file)
    predict_before = _predict_ok(config.host, config.port) if ready_after_start["ready"] else {"ok": False}

    crashed_pid = pid_status_after_start.get("pid")
    crash_event = {"status": "skipped", "reason": "pid_missing"}
    if isinstance(crashed_pid, int):
        crash_event = _kill_process(crashed_pid)
    time.sleep(1.0)
    stale_after_crash = read_pid_status(config.pid_file)
    restart_after_crash = _run_script(start_script, cwd=".", timeout=config.readiness_timeout_seconds + 30.0)
    ready_after_recovery = _wait_http_ready(config.host, config.port, timeout_seconds=config.readiness_timeout_seconds)
    pid_status_after_recovery = read_pid_status(config.pid_file)
    predict_after_recovery = _predict_ok(config.host, config.port) if ready_after_recovery["ready"] else {"ok": False}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        occupied_host, occupied_port = sock.getsockname()
        occupied_available = port_available(occupied_host, occupied_port)

    final_stop = _run_script(stop_script, cwd=".", timeout=90.0)
    final_pid_status = read_pid_status(config.pid_file)
    stage_gates = {
        "daemon_bundle_valid": validation["passes_stage_gate"],
        "log_rotation": rotation_before_start.get("rotated") is True,
        "start_ok": start["returncode"] == 0 and ready_after_start["ready"],
        "predict_before_crash": predict_before.get("ok") is True,
        "crash_simulated": crash_event.get("status") == "ok",
        "stale_pid_detected": stale_after_crash.get("stale") is True,
        "recovery_start_ok": restart_after_crash["returncode"] == 0 and ready_after_recovery["ready"],
        "new_pid_after_recovery": isinstance(pid_status_after_recovery.get("pid"), int)
        and pid_status_after_recovery.get("pid") != crashed_pid,
        "predict_after_recovery": predict_after_recovery.get("ok") is True,
        "port_occupied_detected": occupied_available is False,
        "final_stop_ok": final_stop["returncode"] == 0 and final_pid_status.get("exists") is False,
    }
    summary = {
        "stage": "stage57_daemon_recovery",
        "real": real,
        "config": asdict(config),
        "manifest": manifest,
        "validation": validation,
        "pre_stop": pre_stop,
        "rotation_before_start": rotation_before_start,
        "start": start,
        "ready_after_start": ready_after_start,
        "pid_status_after_start": pid_status_after_start,
        "predict_before": predict_before,
        "crash_event": crash_event,
        "stale_after_crash": stale_after_crash,
        "restart_after_crash": restart_after_crash,
        "ready_after_recovery": ready_after_recovery,
        "pid_status_after_recovery": pid_status_after_recovery,
        "predict_after_recovery": predict_after_recovery,
        "occupied_port_available": occupied_available,
        "final_stop": final_stop,
        "final_pid_status": final_pid_status,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    return summary
