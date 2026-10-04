from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import time
from typing import Any

from .stage51_runtime_management import build_stage51_fake_service
from .stage55_service_deployment import (
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE55_OUTPUT_DIR,
    Stage55DeploymentConfig,
    Stage55ServiceClient,
    _wait_ready,
)
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage56_daemon_operations")


@dataclass(frozen=True)
class Stage56DaemonConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    preferred_device: str = "cuda"
    max_length: int = 384
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json"
    centroid_bundle: str = (
        "artifacts/civilization/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    )
    model_path: str = str(DEFAULT_MODEL_PATH)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    log_dir: str = str(DEFAULT_LOG_DIR)
    pid_file: str = str(DEFAULT_LOG_DIR / "stage56_daemon.pid")
    log_file: str = str(DEFAULT_LOG_DIR / "stage56_daemon.log")
    python_bin: str = ".venv/bin/python"
    readiness_timeout_seconds: float = 240.0
    stop_timeout_seconds: float = 30.0
    max_log_bytes: int = 32 * 1024 * 1024
    log_backup_count: int = 5
    offline: bool = True


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_status(pid_file: str | Path) -> dict[str, Any]:
    path = Path(pid_file)
    if not path.exists():
        return {"pid_file": str(path), "exists": False, "running": False, "pid": None, "stale": False}
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except Exception as error:
        return {
            "pid_file": str(path),
            "exists": True,
            "running": False,
            "pid": None,
            "stale": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    running = process_running(pid)
    return {"pid_file": str(path), "exists": True, "running": running, "pid": pid, "stale": not running}


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def rotate_log_file(log_file: str | Path, *, max_bytes: int, backup_count: int) -> dict[str, Any]:
    path = Path(log_file)
    if backup_count < 1:
        raise ValueError("backup_count must be >= 1")
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    if not path.exists():
        return {"rotated": False, "reason": "missing", "log_file": str(path)}
    size = path.stat().st_size
    if size <= max_bytes:
        return {"rotated": False, "reason": "below_threshold", "log_file": str(path), "bytes": size}
    oldest = path.with_name(path.name + f".{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for index in range(backup_count - 1, 0, -1):
        source = path.with_name(path.name + f".{index}")
        target = path.with_name(path.name + f".{index + 1}")
        if source.exists():
            source.rename(target)
    path.rename(path.with_name(path.name + ".1"))
    path.write_text("", encoding="utf-8")
    return {"rotated": True, "log_file": str(path), "bytes": size, "backup_count": backup_count}


def _service_command(config: Stage56DaemonConfig) -> str:
    env_prefix = "env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 " if config.offline else "env PYTHONUNBUFFERED=1 "
    parts = [
        config.python_bin,
        "-m",
        "civilization.engine.stages.run_qwen3_stage55_service_deployment",
        "--serve",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--preferred-device",
        config.preferred_device,
        "--max-length",
        str(config.max_length),
        "--package-manifest",
        config.package_manifest,
        "--centroid-bundle",
        config.centroid_bundle,
        "--model-path",
        config.model_path,
    ]
    return env_prefix + " ".join(_q(part) for part in parts)


def write_stage56_daemon_bundle(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage56DaemonConfig | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    cfg = config or Stage56DaemonConfig(output_dir=str(output))
    scripts = output / "scripts"
    log_dir = Path(cfg.log_dir)
    pid_file = Path(cfg.pid_file)
    log_file = Path(cfg.log_file)
    command = _service_command(cfg)
    rotate_cmd = (
        f"{_q(cfg.python_bin)} -m civilization.engine.stages.run_qwen3_stage56_daemon_operations "
        f"--rotate-log --log-file {_q(log_file)} --max-log-bytes {cfg.max_log_bytes} "
        f"--log-backup-count {cfg.log_backup_count}"
    )
    port_check = (
        f"{_q(cfg.python_bin)} -m civilization.engine.stages.run_qwen3_stage56_daemon_operations "
        f"--check-port --host {_q(cfg.host)} --port {cfg.port}"
    )
    start_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
mkdir -p {_q(log_dir)}
if [ -f {_q(pid_file)} ]; then
  pid="$(cat {_q(pid_file)} 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "{{\\"status\\":\\"already_running\\",\\"pid\\":$pid}}"
    exit 0
  fi
  rm -f {_q(pid_file)}
fi
{port_check} >/dev/null
{rotate_cmd} >/dev/null
nohup {command} > {_q(log_file)} 2>&1 &
echo "$!" > {_q(pid_file)}
for _ in $(seq 1 {int(cfg.readiness_timeout_seconds)}); do
  if curl -fsS http://{cfg.host}:{cfg.port}/health >/dev/null 2>&1; then
    echo "{{\\"status\\":\\"started\\",\\"pid\\":$(cat {_q(pid_file)}),\\"port\\":{cfg.port}}}"
    exit 0
  fi
  if ! kill -0 "$(cat {_q(pid_file)})" 2>/dev/null; then
    echo "{{\\"status\\":\\"error\\",\\"error\\":\\"process_exited_before_ready\\",\\"log\\":\\"{log_file}\\"}}"
    exit 1
  fi
  sleep 1
done
echo "{{\\"status\\":\\"error\\",\\"error\\":\\"readiness_timeout\\",\\"pid\\":$(cat {_q(pid_file)}),\\"log\\":\\"{log_file}\\"}}"
exit 1
"""
    stop_script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ ! -f {_q(pid_file)} ]; then
  echo "{{\\"status\\":\\"ok\\",\\"stopped\\":false,\\"reason\\":\\"pid_missing\\"}}"
  exit 0
fi
pid="$(cat {_q(pid_file)})"
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f {_q(pid_file)}
  echo "{{\\"status\\":\\"ok\\",\\"stopped\\":false,\\"reason\\":\\"stale_pid\\",\\"pid\\":$pid}}"
  exit 0
fi
kill "$pid"
for _ in $(seq 1 {int(cfg.stop_timeout_seconds)}); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f {_q(pid_file)}
    echo "{{\\"status\\":\\"ok\\",\\"stopped\\":true,\\"pid\\":$pid}}"
    exit 0
  fi
  sleep 1
done
kill -9 "$pid" 2>/dev/null || true
rm -f {_q(pid_file)}
echo "{{\\"status\\":\\"forced\\",\\"stopped\\":true,\\"pid\\":$pid}}"
"""
    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
{_q(cfg.python_bin)} -m civilization.engine.stages.run_qwen3_stage56_daemon_operations --pid-status --pid-file {_q(pid_file)}
curl -fsS http://{cfg.host}:{cfg.port}/health || true
echo
"""
    restart_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./stop_stage56_daemon.sh
./start_stage56_daemon.sh
"""
    reload_script = f"""#!/usr/bin/env bash
set -euo pipefail
curl -fsS -X POST http://{cfg.host}:{cfg.port}/admin/drain
echo
curl -fsS -X POST http://{cfg.host}:{cfg.port}/admin/reload
echo
curl -fsS -X POST http://{cfg.host}:{cfg.port}/admin/resume
echo
"""
    tail_script = f"""#!/usr/bin/env bash
set -euo pipefail
tail -f {_q(log_file)}
"""
    readme = f"""# Stage56 Daemon Operations

Stage56 hardens the Stage55 resident service into an operational daemon bundle.

## Start

```bash
{scripts / "start_stage56_daemon.sh"}
```

The start script:

- clears stale PID files,
- fails fast if port `{cfg.port}` is occupied,
- rotates `{log_file}` before starting,
- starts the service with `nohup`,
- writes `{pid_file}`,
- waits for `/health`.

## Stop / Restart / Status

```bash
{scripts / "stop_stage56_daemon.sh"}
{scripts / "restart_stage56_daemon.sh"}
{scripts / "status_stage56_daemon.sh"}
```

## Reload Adapter Runtime

```bash
{scripts / "reload_stage56_daemon.sh"}
```

## Logs

```bash
{scripts / "tail_stage56_log.sh"}
```
"""
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(output / "daemon_config.json", asdict(cfg))
    _write_executable(scripts / "start_stage56_daemon.sh", start_script)
    _write_executable(scripts / "stop_stage56_daemon.sh", stop_script)
    _write_executable(scripts / "status_stage56_daemon.sh", status_script)
    _write_executable(scripts / "restart_stage56_daemon.sh", restart_script)
    _write_executable(scripts / "reload_stage56_daemon.sh", reload_script)
    _write_executable(scripts / "tail_stage56_log.sh", tail_script)
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "stage": "stage56_daemon_operations",
        "config": asdict(cfg),
        "service_command": command,
        "files": {
            "daemon_config": str(output / "daemon_config.json"),
            "readme": str(output / "README.md"),
            "start_script": str(scripts / "start_stage56_daemon.sh"),
            "stop_script": str(scripts / "stop_stage56_daemon.sh"),
            "status_script": str(scripts / "status_stage56_daemon.sh"),
            "restart_script": str(scripts / "restart_stage56_daemon.sh"),
            "reload_script": str(scripts / "reload_stage56_daemon.sh"),
            "tail_script": str(scripts / "tail_stage56_log.sh"),
        },
        "operational_contract": {
            "background_mode": "nohup",
            "stale_pid_cleanup": True,
            "port_preflight": True,
            "log_rotation": True,
            "readiness_wait": True,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        },
    }
    _json_dump(output / "daemon_manifest.json", manifest)
    return manifest


def validate_stage56_daemon_bundle(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "daemon_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {name: Path(path) for name, path in manifest["files"].items()}
    start_body = files["start_script"].read_text(encoding="utf-8")
    stop_body = files["stop_script"].read_text(encoding="utf-8")
    checks = {
        "manifest_exists": manifest_path.exists(),
        "required_files_exist": all(path.exists() for path in files.values()),
        "scripts_executable": all(os.access(path, os.X_OK) for key, path in files.items() if key.endswith("_script")),
        "uses_nohup": "nohup" in start_body,
        "clears_stale_pid": "rm -f" in start_body and "pid" in start_body,
        "checks_port": "--check-port" in start_body,
        "rotates_log": "--rotate-log" in start_body,
        "waits_ready": "/health" in start_body,
        "stop_idempotent": "pid_missing" in stop_body and "stale_pid" in stop_body,
        "contract_background": manifest["operational_contract"]["background_mode"] == "nohup",
    }
    result = {"stage": "stage56_daemon_validation", "checks": checks, "passes_stage_gate": all(checks.values())}
    _json_dump(output / "daemon_validation.json", result)
    return result


def run_stage56_fake_daemon_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR, port: int = 0) -> dict[str, Any]:
    output = Path(output_dir)
    cfg = Stage56DaemonConfig(
        output_dir=str(output),
        log_dir=str(output / "logs"),
        pid_file=str(output / "logs" / "stage56.pid"),
        log_file=str(output / "logs" / "stage56.log"),
        port=8765 if port == 0 else port,
        max_log_bytes=8,
        log_backup_count=2,
    )
    manifest = write_stage56_daemon_bundle(output_dir=output, config=cfg)
    validation = validate_stage56_daemon_bundle(output_dir=output)
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("0123456789abcdef", encoding="utf-8")
    rotate = rotate_log_file(log_path, max_bytes=cfg.max_log_bytes, backup_count=cfg.log_backup_count)
    stale_pid_path = output / "logs" / "stale.pid"
    stale_pid_path.write_text("99999999", encoding="utf-8")
    stale_status = read_pid_status(stale_pid_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        occupied_host, occupied_port = sock.getsockname()
        occupied_port_available = port_available(occupied_host, occupied_port)
    service = build_stage51_fake_service(port=port)
    server = service.start_background()
    host, actual_port = server.server_address
    client = Stage55ServiceClient(f"http://{host}:{actual_port}", timeout=10.0)
    try:
        health = _wait_ready(client, timeout_seconds=10.0)
        predict = client.predict(
            {
                "id": "stage56-smoke",
                "text": "Evaluate operation.",
                "memory_items": ["memory"],
                "rule_items": ["rule"],
                "state_values": [1.0, 0.0, 0.5],
                "answer_options": ["approve", "reject"],
                "task_name": "operation_decision",
                "seed": 202,
                "controls": ["full", "adapter_disabled"],
            }
        )
        reload_response = client.reload()
    finally:
        service.shutdown()
    stage_gates = {
        "bundle_valid": validation["passes_stage_gate"],
        "log_rotated": rotate.get("rotated") is True and log_path.with_name(log_path.name + ".1").exists(),
        "stale_pid_detected": stale_status.get("stale") is True,
        "port_occupied_detected": occupied_port_available is False,
        "fake_service_ready": health.get("ready") is True,
        "predict_ok": predict.get("status") == "ok" and all(row.get("status") == "ok" for row in predict.get("rows", [])),
        "reload_ok": reload_response.get("event", {}).get("status") == "ok",
    }
    summary = {
        "stage": "stage56_daemon_operations_smoke",
        "manifest": manifest,
        "validation": validation,
        "rotate": rotate,
        "stale_pid_status": stale_status,
        "occupied_port_available": occupied_port_available,
        "health": health,
        "predict": predict,
        "reload_response": reload_response,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    return summary
