from __future__ import annotations

from dataclasses import asdict, dataclass, field
from http import HTTPStatus
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import (
    Stage51ManagedInferenceService,
    Stage51RuntimeDescriptor,
    Stage51VersionedFakeRuntime,
)
from .stage52_real_runtime_reload_stress import build_stage52_real_managed_service
from experiments.civilization_transformer_qwen3.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage59_access_control_audit")
DEFAULT_AUDIT_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage59_audit.jsonl")


@dataclass(frozen=True)
class Stage59SecurityConfig:
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "::1", "localhost")
    bearer_token: str | None = None
    require_token_for_predict: bool = False
    require_token_for_admin: bool = True
    require_token_for_metrics: bool = False
    audit_log_path: str = str(DEFAULT_AUDIT_LOG)
    redact_text_max_chars: int = 0
    hash_salt: str = "stage59-local"


@dataclass
class Stage59SecurityMetrics:
    access_allowed: int = 0
    access_denied: int = 0
    auth_failures: int = 0
    audit_events: int = 0
    last_denial_reason: str | None = None
    last_audit_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def _hash_text(value: str, *, salt: str) -> str:
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:16]


def redact_payload(payload: Any, *, config: Stage59SecurityConfig) -> Any:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "text" and isinstance(value, str):
            redacted[key] = {
                "chars": len(value),
                "sha256_16": _hash_text(value, salt=config.hash_salt),
                "preview": value[: config.redact_text_max_chars] if config.redact_text_max_chars > 0 else "",
            }
        elif key in {"memory_items", "rule_items"} and isinstance(value, list):
            redacted[key] = [
                {"chars": len(str(item)), "sha256_16": _hash_text(str(item), salt=config.hash_salt)}
                for item in value
            ]
        elif key == "answer_options" and isinstance(value, list):
            redacted[key] = [{"chars": len(str(item))} for item in value]
        elif key in {"state_values", "task_name", "seed", "controls", "readouts", "id"}:
            redacted[key] = value
        else:
            redacted[key] = {"type": type(value).__name__}
    return redacted


class Stage59AccessControlledService(Stage51ManagedInferenceService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
    ) -> None:
        super().__init__(runtime_factory=runtime_factory, descriptor=descriptor, config=config)
        self.security = security
        self.security_metrics = Stage59SecurityMetrics()

    def _client_host_allowed(self, host: str) -> bool:
        return host in set(self.security.allowed_hosts)

    def _path_requires_token(self, path: str) -> bool:
        if self.security.bearer_token is None:
            return False
        if path.startswith("/admin/"):
            return self.security.require_token_for_admin
        if path in {"/v1/predict", "/v1/batch"}:
            return self.security.require_token_for_predict
        if path == "/metrics":
            return self.security.require_token_for_metrics
        return False

    def _token_valid(self, authorization: str | None) -> bool:
        if self.security.bearer_token is None:
            return True
        return authorization == f"Bearer {self.security.bearer_token}"

    def _audit(self, event: dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **event}
        path = Path(self.security.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.security_metrics.audit_events += 1
        except Exception as error:
            self.security_metrics.last_audit_error = str(error)

    def security_status(self) -> dict[str, Any]:
        return {
            "allowed_hosts": list(self.security.allowed_hosts),
            "bearer_token_configured": self.security.bearer_token is not None,
            "require_token_for_predict": self.security.require_token_for_predict,
            "require_token_for_admin": self.security.require_token_for_admin,
            "require_token_for_metrics": self.security.require_token_for_metrics,
            "audit_log_path": self.security.audit_log_path,
            "metrics": self.security_metrics.snapshot(),
        }

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage59_access_control_audit"
        payload["security"] = self.security_status()
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage59Handler(base_handler):
            server_version = "Stage59SecureQwenService/1.0"

            def _deny(self, status: HTTPStatus, reason: str) -> None:
                service.security_metrics.access_denied += 1
                service.security_metrics.last_denial_reason = reason
                if status == HTTPStatus.UNAUTHORIZED:
                    service.security_metrics.auth_failures += 1
                service._audit(
                    {
                        "event": "access_denied",
                        "path": self.path,
                        "method": self.command,
                        "client_host": self.client_address[0],
                        "status": int(status),
                        "reason": reason,
                    }
                )
                self._send_json(status, {"status": "error", "error": reason})

            def _check_access(self, payload: Any | None = None) -> bool:
                client_host = self.client_address[0]
                if not service._client_host_allowed(client_host):
                    self._deny(HTTPStatus.FORBIDDEN, "client host is not allowed")
                    return False
                if service._path_requires_token(self.path) and not service._token_valid(self.headers.get("Authorization")):
                    self._deny(HTTPStatus.UNAUTHORIZED, "missing or invalid bearer token")
                    return False
                service.security_metrics.access_allowed += 1
                service._audit(
                    {
                        "event": "access_allowed",
                        "path": self.path,
                        "method": self.command,
                        "client_host": client_host,
                        "payload": redact_payload(payload, config=service.security) if payload is not None else None,
                    }
                )
                return True

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if not self._check_access():
                    return
                if self.path == "/admin/security":
                    self._send_json(HTTPStatus.OK, {"status": "ok", "security": service.security_status()})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                if self.path in {"/v1/predict", "/v1/batch"}:
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                        return
                    if not self._check_access(payload):
                        return
                    try:
                        if self.path == "/v1/predict":
                            rows = service.predict_record(payload)
                            self._send_json(HTTPStatus.OK, {"status": "ok", "rows": rows, "metrics": service.metrics.snapshot()})
                            return
                        records = payload.get("requests") if isinstance(payload, dict) else None
                        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                            raise ValueError("batch payload must contain a requests object array")
                        self._send_json(HTTPStatus.OK, service.predict_batch(records))
                        return
                    except Exception as error:
                        service.metrics.failures += 1
                        service.metrics.last_error = str(error)
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                        return
                if not self._check_access():
                    return
                super().do_POST()

        return Stage59Handler


def build_stage59_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
) -> Stage59AccessControlledService:
    return Stage59AccessControlledService(
        runtime_factory=Stage51VersionedFakeRuntime,
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage59-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
    )


def build_stage59_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage59AccessControlledService:
    base = build_stage52_real_managed_service(
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
        port=port,
    )
    return Stage59AccessControlledService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - intentionally reusing Stage52 shared-backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
    )


def _post_json(url: str, payload: dict[str, Any] | None = None, *, token: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, token: str | None = None) -> dict[str, Any]:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke client
        return json.loads(response.read().decode("utf-8"))


def _payload() -> dict[str, Any]:
    return {
        "id": "stage59-smoke",
        "text": "Sensitive long user text should not appear in audit logs.",
        "memory_items": ["memory item with potentially sensitive detail"],
        "rule_items": ["rule item with potentially sensitive detail"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
        "controls": ["full", "adapter_disabled"],
    }


def run_stage59_security_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_log = output / "audit.jsonl"
    token = "stage59-test-token"
    security = Stage59SecurityConfig(
        bearer_token=token,
        require_token_for_predict=True,
        require_token_for_admin=True,
        require_token_for_metrics=True,
        audit_log_path=str(audit_log),
    )
    service = build_stage59_fake_service(port=0, security=security)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    unauthorized_predict: dict[str, Any]
    unauthorized_admin: dict[str, Any]
    try:
        health = _get_json(f"{base}/health")
        try:
            _post_json(f"{base}/v1/predict", _payload())
            unauthorized_predict = {"raised": False}
        except HTTPError as error:
            unauthorized_predict = {"raised": True, "code": error.code}
        predict = _post_json(f"{base}/v1/predict", _payload(), token=token)
        try:
            _post_json(f"{base}/admin/reload")
            unauthorized_admin = {"raised": False}
        except HTTPError as error:
            unauthorized_admin = {"raised": True, "code": error.code}
        reload_response = _post_json(f"{base}/admin/reload", token=token)
        security_status = _get_json(f"{base}/admin/security", token=token)
    finally:
        service.shutdown()
    audit_rows = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    audit_text = audit_log.read_text(encoding="utf-8")
    stage_gates = {
        "health_public": health.get("ready") is True,
        "predict_requires_token": unauthorized_predict.get("code") == 401,
        "admin_requires_token": unauthorized_admin.get("code") == 401,
        "predict_with_token_ok": predict.get("status") == "ok",
        "reload_with_token_ok": reload_response.get("event", {}).get("status") == "ok",
        "audit_written": len(audit_rows) >= 4,
        "audit_redacts_text": "Sensitive long user text" not in audit_text,
        "security_status_available": security_status.get("security", {}).get("bearer_token_configured") is True,
    }
    summary = {
        "stage": "stage59_access_control_audit_smoke",
        "health": health,
        "unauthorized_predict": unauthorized_predict,
        "predict": predict,
        "unauthorized_admin": unauthorized_admin,
        "reload_response": reload_response,
        "security_status": security_status,
        "audit_rows": audit_rows,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
