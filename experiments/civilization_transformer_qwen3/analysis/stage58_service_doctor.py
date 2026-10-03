from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
from typing import Any

from .stage55_service_deployment import Stage55ServiceClient
from .stage56_daemon_operations import (
    Stage56DaemonConfig,
    port_available,
    read_pid_status,
    validate_stage56_daemon_bundle,
    write_stage56_daemon_bundle,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage58_service_doctor")


@dataclass(frozen=True)
class Stage58DoctorConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    preferred_device: str = "cuda"
    max_length: int = 384
    repo_root: str = "/home/yike/AoNeb-01"
    python_bin: str = ".venv/bin/python"
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B"
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json"
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    )
    daemon_output_dir: str = "experiments/civilization_transformer_qwen3/artifacts/stage56_daemon_operations"
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    log_dir: str = "experiments/civilization_transformer_qwen3/artifacts/logs"
    pid_file: str = "experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.pid"
    log_file: str = "experiments/civilization_transformer_qwen3/artifacts/logs/stage56_daemon.log"
    systemd_unit_name: str = "aoneb-qwen3-civilization.service"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _path_status(path: str | Path, *, must_exist: bool = True) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "bytes": p.stat().st_size if p.exists() and p.is_file() else None,
        "ok": p.exists() if must_exist else True,
    }


def _python_import_check(python_bin: str, *, cwd: str | Path) -> dict[str, Any]:
    command = [
        python_bin,
        "-c",
        "import torch, transformers; print('ok')",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as error:
        return {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _cuda_check(python_bin: str, *, cwd: str | Path) -> dict[str, Any]:
    command = [
        python_bin,
        "-c",
        "import torch, json; print(json.dumps({'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}))",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as error:
        return {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    payload: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            payload = {"raw_stdout": completed.stdout.strip()}
    return {
        "ok": completed.returncode == 0 and payload.get("cuda_available") is True,
        "returncode": completed.returncode,
        "payload": payload,
        "stderr": completed.stderr.strip(),
    }


def _health_check(host: str, port: int) -> dict[str, Any]:
    client = Stage55ServiceClient(f"http://{host}:{port}", timeout=5.0)
    try:
        health = client.health()
    except Exception as error:
        return {"reachable": False, "ok": True, "error_type": type(error).__name__, "error": str(error)}
    return {"reachable": True, "ok": health.get("ready") is True, "health": health}


def _port_probe(host: str, port: int, *, pid_running: bool) -> dict[str, Any]:
    available = port_available(host, port)
    return {
        "host": host,
        "port": port,
        "available": available,
        "ok": (not available) if pid_running else available,
        "expected": "occupied_by_service" if pid_running else "free",
    }


def _systemctl_available() -> bool:
    return shutil.which("systemctl") is not None


def _daemon_config(config: Stage58DoctorConfig) -> Stage56DaemonConfig:
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
    )


def generate_stage58_runbook(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage58DoctorConfig = Stage58DoctorConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    scripts = output / "scripts"
    daemon_manifest = write_stage56_daemon_bundle(output_dir=config.daemon_output_dir, config=_daemon_config(config))
    unit_path = output / config.systemd_unit_name
    start_script = Path(daemon_manifest["files"]["start_script"])
    stop_script = Path(daemon_manifest["files"]["stop_script"])
    restart_script = Path(daemon_manifest["files"]["restart_script"])
    status_script = Path(daemon_manifest["files"]["status_script"])
    unit_body = f"""[Unit]
Description=AoNeb Qwen3 Civilization Adapter Service
After=network.target

[Service]
Type=forking
WorkingDirectory={config.repo_root}
ExecStart={config.repo_root}/{start_script}
ExecStop={config.repo_root}/{stop_script}
ExecReload={config.repo_root}/{restart_script}
PIDFile={config.repo_root}/{config.pid_file}
Restart=on-failure
RestartSec=5
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    install_script = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/.config/systemd/user"
cp "{unit_path}" "$HOME/.config/systemd/user/{config.systemd_unit_name}"
systemctl --user daemon-reload
systemctl --user enable {config.systemd_unit_name}
echo "installed {config.systemd_unit_name}; start with: systemctl --user start {config.systemd_unit_name}"
"""
    uninstall_script = f"""#!/usr/bin/env bash
set -euo pipefail
systemctl --user disable --now {config.systemd_unit_name} 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/{config.systemd_unit_name}"
systemctl --user daemon-reload
echo "uninstalled {config.systemd_unit_name}"
"""
    recover_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {config.repo_root}
{stop_script} || true
{start_script}
{status_script}
"""
    doctor_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {config.repo_root}
{config.python_bin} -m experiments.civilization_transformer_qwen3.analysis.run_qwen3_stage58_service_doctor --doctor
"""
    output.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_body, encoding="utf-8")
    _write_executable(scripts / "install_user_systemd.sh", install_script)
    _write_executable(scripts / "uninstall_user_systemd.sh", uninstall_script)
    _write_executable(scripts / "recover_stage58_service.sh", recover_script)
    _write_executable(scripts / "doctor_stage58_service.sh", doctor_script)
    readme = f"""# Stage58 Service Doctor and Recovery Runbook

## Doctor

```bash
{scripts / "doctor_stage58_service.sh"}
```

## Manual Recovery

```bash
{scripts / "recover_stage58_service.sh"}
```

## Optional user systemd install

This is generated as a template. It is not installed automatically.

```bash
{scripts / "install_user_systemd.sh"}
systemctl --user start {config.systemd_unit_name}
systemctl --user status {config.systemd_unit_name}
```

## Uninstall

```bash
{scripts / "uninstall_user_systemd.sh"}
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "stage": "stage58_service_doctor",
        "config": asdict(config),
        "daemon_manifest": daemon_manifest,
        "files": {
            "readme": str(output / "README.md"),
            "systemd_unit": str(unit_path),
            "install_script": str(scripts / "install_user_systemd.sh"),
            "uninstall_script": str(scripts / "uninstall_user_systemd.sh"),
            "recover_script": str(scripts / "recover_stage58_service.sh"),
            "doctor_script": str(scripts / "doctor_stage58_service.sh"),
        },
        "installed_systemd": False,
    }
    _json_dump(output / "runbook_manifest.json", manifest)
    return manifest


def run_stage58_doctor(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage58DoctorConfig = Stage58DoctorConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runbook = generate_stage58_runbook(output_dir=output, config=config)
    daemon_validation = validate_stage56_daemon_bundle(output_dir=config.daemon_output_dir)
    pid = read_pid_status(config.pid_file)
    checks = {
        "repo_root": _path_status(config.repo_root),
        "python_bin": _path_status(Path(config.repo_root) / config.python_bin),
        "model_path": _path_status(config.model_path),
        "package_manifest": _path_status(Path(config.repo_root) / config.package_manifest),
        "centroid_bundle": _path_status(Path(config.repo_root) / config.centroid_bundle),
        "daemon_bundle": daemon_validation,
        "pid_status": pid,
        "port": _port_probe(config.host, config.port, pid_running=pid.get("running") is True),
        "health": _health_check(config.host, config.port),
        "python_imports": _python_import_check(config.python_bin, cwd=config.repo_root),
        "cuda": _cuda_check(config.python_bin, cwd=config.repo_root),
        "systemctl_available": {"ok": True, "available": _systemctl_available()},
        "runbook": {"ok": True, "manifest": runbook},
    }
    remediation: list[str] = []
    if not checks["model_path"]["ok"]:
        remediation.append("Fix Qwen3 model_path or mount the model directory.")
    if not checks["python_imports"]["ok"]:
        remediation.append("Recreate .venv or install torch/transformers dependencies.")
    if not checks["cuda"]["ok"]:
        remediation.append("Check NVIDIA driver, WSL CUDA, or run with preferred_device=cpu for diagnosis.")
    if not checks["daemon_bundle"]["passes_stage_gate"]:
        remediation.append("Regenerate Stage56 daemon bundle.")
    if checks["pid_status"].get("stale"):
        remediation.append("Run stop_stage56_daemon.sh to clear stale PID, then start_stage56_daemon.sh.")
    if not checks["port"]["ok"]:
        remediation.append("Free the configured port or stop the stale service.")
    hard_checks = {
        "repo_root": checks["repo_root"]["ok"],
        "python_bin": checks["python_bin"]["ok"],
        "model_path": checks["model_path"]["ok"],
        "package_manifest": checks["package_manifest"]["ok"],
        "centroid_bundle": checks["centroid_bundle"]["ok"],
        "daemon_bundle": checks["daemon_bundle"]["passes_stage_gate"],
        "python_imports": checks["python_imports"]["ok"],
        "cuda": checks["cuda"]["ok"],
        "port_consistent": checks["port"]["ok"],
    }
    summary = {
        "stage": "stage58_service_doctor",
        "config": asdict(config),
        "checks": checks,
        "hard_checks": hard_checks,
        "remediation": remediation,
        "passes_stage_gate": all(hard_checks.values()),
    }
    _json_dump(output / "doctor_summary.json", summary)
    return summary
