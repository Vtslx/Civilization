from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapter_benchmark import DEFAULT_MODEL_PATH
from .stage49_persistent_inference_service import DEFAULT_CENTROID_BUNDLE, DEFAULT_PACKAGE_MANIFEST
from .stage51_runtime_management import build_stage51_fake_service
from .stage52_real_runtime_reload_stress import build_stage52_real_managed_service


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage55_service_deployment")
DEFAULT_LOG_DIR = Path("artifacts/civilization/logs")


@dataclass(frozen=True)
class Stage55DeploymentConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    preferred_device: str = "cuda"
    max_length: int = 384
    package_manifest: str = str(DEFAULT_PACKAGE_MANIFEST)
    centroid_bundle: str = str(DEFAULT_CENTROID_BUNDLE)
    model_path: str = str(DEFAULT_MODEL_PATH)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    log_dir: str = str(DEFAULT_LOG_DIR)
    pid_file: str = str(DEFAULT_LOG_DIR / "stage55_service.pid")
    log_file: str = str(DEFAULT_LOG_DIR / "stage55_service.log")
    request_timeout_seconds: float = 240.0
    python_bin: str = ".venv/bin/python"
    offline: bool = True


class Stage55ServiceClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        with urlopen(f"{self.base_url}{path}", timeout=self.timeout) as response:  # noqa: S310 - local client
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - local client
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def ready(self) -> dict[str, Any]:
        return self._get("/ready")

    def metrics(self) -> dict[str, Any]:
        return self._get("/metrics")

    def status(self) -> dict[str, Any]:
        return self._get("/admin/status")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/predict", payload)

    def batch(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post("/v1/batch", {"requests": payloads})

    def drain(self) -> dict[str, Any]:
        return self._post("/admin/drain")

    def resume(self) -> dict[str, Any]:
        return self._post("/admin/resume")

    def reload(self) -> dict[str, Any]:
        return self._post("/admin/reload")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _service_command(config: Stage55DeploymentConfig) -> str:
    parts = [
        config.python_bin,
        "-m",
        "civilization.engine.stages.run_qwen3_stage55_service_deployment",
        "--serve",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--package-manifest",
        config.package_manifest,
        "--centroid-bundle",
        config.centroid_bundle,
        "--model-path",
        config.model_path,
        "--preferred-device",
        config.preferred_device,
        "--max-length",
        str(config.max_length),
    ]
    return " ".join(_shell_quote(part) for part in parts)


def write_stage55_deployment_bundle(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: Stage55DeploymentConfig | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    cfg = config or Stage55DeploymentConfig(output_dir=str(output))
    log_dir = Path(cfg.log_dir)
    pid_file = Path(cfg.pid_file)
    log_file = Path(cfg.log_file)
    scripts_dir = output / "scripts"
    env_prefix = "env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 " if cfg.offline else "env PYTHONUNBUFFERED=1 "
    command = env_prefix + _service_command(cfg)
    start_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
mkdir -p {_shell_quote(log_dir)}
if [ -f {_shell_quote(pid_file)} ] && kill -0 "$(cat {_shell_quote(pid_file)})" 2>/dev/null; then
  echo "stage55 service already running: $(cat {_shell_quote(pid_file)})"
  exit 0
fi
nohup {command} > {_shell_quote(log_file)} 2>&1 &
echo "$!" > {_shell_quote(pid_file)}
echo "stage55 service pid=$(cat {_shell_quote(pid_file)}) log={log_file}"
"""
    stop_script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ ! -f {_shell_quote(pid_file)} ]; then
  echo "stage55 pid file missing"
  exit 0
fi
pid="$(cat {_shell_quote(pid_file)})"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f {_shell_quote(pid_file)}
echo "stage55 service stopped"
"""
    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
if [ -f {_shell_quote(pid_file)} ]; then
  pid="$(cat {_shell_quote(pid_file)})"
  ps -p "$pid" -o pid,ppid,etime,cmd || true
else
  echo "stage55 pid file missing"
fi
curl -fsS http://{cfg.host}:{cfg.port}/health || true
echo
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
    example_payload = {
        "id": "stage55-example",
        "text": "A controller must decide whether to approve the operation using the supplied evidence.",
        "memory_items": ["The operational memory supports approval."],
        "rule_items": ["Use grounded memory evidence unless a stronger rule overrides it."],
        "state_values": [0.9, 0.1, 0.8],
        "answer_options": ["Approve the action.", "Reject the action."],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "no_memory_path", "adapter_disabled"],
        "readouts": ["projected_delta", "raw_full_hidden_fixed_centroid", "projected_full_hidden"],
    }
    predict_script = f"""#!/usr/bin/env bash
set -euo pipefail
curl -fsS -X POST http://{cfg.host}:{cfg.port}/v1/predict \\
  -H 'Content-Type: application/json' \\
  --data-binary @{_shell_quote(output / "example_request.json")}
echo
"""
    readme = f"""# Stage55 Local Service Deployment

This package wraps the Stage49-54 resident Qwen3 Civilization service into an operational local service.

## Start

```bash
{scripts_dir / "start_stage55_service.sh"}
```

The start script uses `nohup`, writes PID to `{pid_file}`, and writes logs to `{log_file}`.

## Monitor

```bash
{scripts_dir / "status_stage55_service.sh"}
tail -f {log_file}
```

## Predict

```bash
{scripts_dir / "predict_example.sh"}
```

## Reload

```bash
{scripts_dir / "reload_stage55_service.sh"}
```

## Stop

```bash
{scripts_dir / "stop_stage55_service.sh"}
```

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /admin/status`
- `POST /v1/predict`
- `POST /v1/batch`
- `POST /admin/drain`
- `POST /admin/reload`
- `POST /admin/resume`

Qwen3 remains frozen. Service reloads replace adapter/readout runtime objects while sharing the loaded Qwen3 backend.
"""
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(output / "service_config.json", asdict(cfg))
    _json_dump(output / "example_request.json", example_payload)
    _write_executable(scripts_dir / "start_stage55_service.sh", start_script)
    _write_executable(scripts_dir / "stop_stage55_service.sh", stop_script)
    _write_executable(scripts_dir / "status_stage55_service.sh", status_script)
    _write_executable(scripts_dir / "reload_stage55_service.sh", reload_script)
    _write_executable(scripts_dir / "predict_example.sh", predict_script)
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "stage": "stage55_service_deployment",
        "config": asdict(cfg),
        "service_command": command,
        "files": {
            "service_config": str(output / "service_config.json"),
            "example_request": str(output / "example_request.json"),
            "readme": str(output / "README.md"),
            "start_script": str(scripts_dir / "start_stage55_service.sh"),
            "stop_script": str(scripts_dir / "stop_stage55_service.sh"),
            "status_script": str(scripts_dir / "status_stage55_service.sh"),
            "reload_script": str(scripts_dir / "reload_stage55_service.sh"),
            "predict_script": str(scripts_dir / "predict_example.sh"),
        },
        "operational_contract": {
            "background_mode": "nohup",
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "qwen_frozen": True,
            "offline_default": cfg.offline,
        },
    }
    _json_dump(output / "deployment_manifest.json", manifest)
    return manifest


def validate_stage55_deployment_bundle(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    required = {name: Path(path) for name, path in files.items()}
    executable_names = {"start_script", "stop_script", "status_script", "reload_script", "predict_script"}
    start_body = required["start_script"].read_text(encoding="utf-8")
    pid_file = str(manifest["operational_contract"]["pid_file"])
    checks = {
        "manifest_exists": manifest_path.exists(),
        "required_files_exist": all(path.exists() for path in required.values()),
        "scripts_executable": all(os.access(required[name], os.X_OK) for name in executable_names),
        "start_uses_nohup": "nohup" in start_body,
        "start_writes_pid": pid_file in start_body,
        "offline_default": manifest["operational_contract"]["offline_default"] is True,
        "qwen_frozen_claim": manifest["operational_contract"]["qwen_frozen"] is True,
    }
    result = {
        "stage": "stage55_deployment_validation",
        "checks": checks,
        "passes_stage_gate": all(checks.values()),
    }
    _json_dump(output / "deployment_validation.json", result)
    return result


def _wait_ready(client: Stage55ServiceClient, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            health = client.health()
            if health.get("ready") is True:
                return health
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise TimeoutError(f"service was not ready before timeout; last_error={last_error}")


def _example_payload() -> dict[str, Any]:
    return {
        "id": "stage55-smoke",
        "text": "Evaluate the operation using memory and rule context.",
        "memory_items": ["memory supports approval"],
        "rule_items": ["rule permits approval"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "no_memory_path", "adapter_disabled"],
    }


def run_stage55_fake_deployment_smoke(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    port: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    config = Stage55DeploymentConfig(output_dir=str(output), port=8765 if port == 0 else port)
    manifest = write_stage55_deployment_bundle(output_dir=output, config=config)
    validation = validate_stage55_deployment_bundle(output_dir=output)
    service = build_stage51_fake_service(port=port)
    server = service.start_background()
    host, actual_port = server.server_address
    client = Stage55ServiceClient(f"http://{host}:{actual_port}", timeout=10.0)
    try:
        health = _wait_ready(client, timeout_seconds=10.0)
        status_before = client.status()
        predict = client.predict(_example_payload())
        drain = client.drain()
        draining_predict = client.predict(_example_payload())
        reload_response = client.reload()
        resume = client.resume()
        predict_after = client.predict(_example_payload())
        metrics = client.metrics()
    finally:
        service.shutdown()
    rows = predict.get("rows", [])
    after_rows = predict_after.get("rows", [])
    stage_gates = {
        "deployment_bundle_valid": validation["passes_stage_gate"],
        "service_ready": health.get("ready") is True,
        "status_has_management": "management" in status_before,
        "predict_ok": predict.get("status") == "ok" and rows and all(row.get("status") == "ok" for row in rows),
        "drain_rejects_prediction": draining_predict.get("rows", [{}])[0].get("error_type") == "ServiceDraining",
        "reload_ok": reload_response.get("event", {}).get("status") == "ok",
        "resume_ok": resume.get("management", {}).get("draining") is False,
        "predict_after_reload_ok": predict_after.get("status") == "ok"
        and after_rows
        and all(row.get("status") == "ok" for row in after_rows),
        "metrics_recorded": metrics.get("requests_received", 0) >= 2,
    }
    summary = {
        "stage": "stage55_service_deployment_smoke",
        "manifest": manifest,
        "validation": validation,
        "health": health,
        "status_before": status_before,
        "predict": predict,
        "drain": drain,
        "draining_predict": draining_predict,
        "reload_response": reload_response,
        "resume": resume,
        "predict_after": predict_after,
        "metrics": metrics,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    _json_dump(output / "summary.json", summary)
    return summary


def serve_stage55_real_service(config: Stage55DeploymentConfig) -> None:
    service = build_stage52_real_managed_service(
        package_manifest=config.package_manifest,
        centroid_bundle=config.centroid_bundle,
        model_path=config.model_path,
        preferred_device=config.preferred_device,
        max_length=config.max_length,
        port=config.port,
    )
    print(
        json.dumps(
            {
                "stage": "stage55_service_deployment",
                "status": "starting",
                "host": config.host,
                "port": config.port,
                "preferred_device": config.preferred_device,
                "max_length": config.max_length,
                "package_manifest": config.package_manifest,
                "centroid_bundle": config.centroid_bundle,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    service.serve_forever()


def stop_pid_file(pid_file: str | Path, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    path = Path(pid_file)
    if not path.exists():
        return {"status": "ok", "message": "pid file missing", "stopped": False}
    pid = int(path.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, 0)
    except OSError:
        path.unlink(missing_ok=True)
        return {"status": "ok", "message": "process not running", "pid": pid, "stopped": False}
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            path.unlink(missing_ok=True)
            return {"status": "ok", "pid": pid, "stopped": True}
        time.sleep(0.2)
    return {"status": "error", "pid": pid, "stopped": False, "error": "process did not exit before timeout"}
