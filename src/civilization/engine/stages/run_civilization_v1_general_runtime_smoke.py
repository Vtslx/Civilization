"""General-runtime production smoke for the provider-agnostic v1 service.

Runs the real Stage49-74 chain on top of the runtime layer in
``civilization.runtimes``, drives the full client surface (health, ready,
capabilities, sync prediction, batch, Orion memory, async job, export package
download), and writes an auditable summary. Any OpenAI-compatible Chat
Completions endpoint can be the decision backend; the run is not tied to one
model family.

Usage:

    CIVILIZATION_SMOKE_API_KEY=... python -m \
      civilization.engine.stages.run_civilization_general_runtime_smoke \
      --provider-base-url https://provider.example.com/v1 \
      --provider-model some-model \
      --serve

``--serve`` keeps the service listening after the smoke and prints a JSON line
with the bound base URL, which is how the TypeScript SDK live test attaches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

from civilization import (
    CivilizationRequest,
    EmbeddedCivilization,
    EmbeddedConfig,
    capabilities_for,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/civilization/v1_general_runtime_smoke"
)


def _request(request_id: str, session_id: str) -> CivilizationRequest:
    return CivilizationRequest(
        request_id=request_id,
        text=(
            "The deployment signature and health checks are valid, and no critical "
            "verification is unresolved. Choose the supported operation."
        ),
        answer_options=("approve", "reject"),
        session_id=session_id,
        task_name="general_runtime_smoke",
        memory_items=("The deployment signature and health checks are valid.",),
        rule_items=("Reject only when a critical verification is unresolved.",),
        state_values=(0.9, 0.1, 0.8),
    )


def _config(
    *,
    provider_base_url: str,
    provider_model: str,
    api_key_env: str,
    output_dir: str | Path,
    allow_insecure_http: bool,
    headers: dict[str, str],
) -> EmbeddedConfig:
    return EmbeddedConfig(
        runtime="provider",
        provider_base_url=provider_base_url,
        provider_model=provider_model,
        provider_api_key_env=api_key_env,
        provider_headers=headers,
        bearer_token_env=None,
        allow_insecure_http=allow_insecure_http,
        state_dir=str(output_dir),
    )


def run_smoke(
    *,
    provider_base_url: str,
    provider_model: str,
    api_key_env: str,
    output_dir: str | Path,
    allow_insecure_http: bool,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    session_id = "general-runtime-smoke"
    config = _config(
        provider_base_url=provider_base_url,
        provider_model=provider_model,
        api_key_env=api_key_env,
        output_dir=output / "state",
        allow_insecure_http=allow_insecure_http,
        headers=headers or {},
    )
    gates: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": {"base_url": provider_base_url, "model": provider_model},
        "api_key_present_in_summary": False,
        "config": {"embedded": asdict(config), "output_dir": str(output)},
        "runtime_capabilities": capabilities_for(config.runtime).to_dict(),
    }
    with EmbeddedCivilization(config) as runtime:
        client = runtime.start()
        summary["gates"] = gates
        gates["ready"] = bool(client.ready().get("ready"))
        capabilities_payload = client._request("GET", "/v1/capabilities")
        summary["capabilities_endpoint"] = capabilities_payload
        gates["capabilities_declared"] = (
            capabilities_payload.get("capabilities", {}).get("adapter_execution") is False
        )

        seed = client.write_memory(
            session_id=session_id,
            memory_system="episodic",
            content="The deployment signature and health checks are valid.",
            summary="deployment verification evidence",
        )
        summary["seed_memory_cell"] = seed
        gates["seed_memory_written"] = bool(seed.get("cell_id"))

        prediction = client.predict(_request("smoke-sync-1", session_id))
        summary["sync_prediction"] = {
            "option_id": prediction.option_id,
            "option_text": prediction.option_text,
            "scores": {name: list(values) for name, values in prediction.scores.items()},
            "trace": dict(prediction.trace),
            "memory_trace": dict(prediction.memory_trace),
        }
        gates["sync_prediction_ok"] = prediction.option_text in {"approve", "reject"}
        gates["production_full_only"] = prediction.trace.get("production_full_mode") is True
        gates["external_runtime_used"] = prediction.trace.get("runtime") == "openai_compatible_chat_completions"
        gates["adapter_claim_is_false"] = (
            prediction.trace.get("adapter_execution") is False
            and prediction.trace.get("hidden_states_available") is False
        )
        memory_trace = dict(prediction.memory_trace)
        gates["memory_retrieved"] = (
            int(memory_trace.get("injected_item_count", 0)) >= 1
            or bool(memory_trace.get("retrieved_cell_ids"))
        )
        gates["memory_session_scoped"] = memory_trace.get("session_id") == session_id

        batch = client.batch([_request("smoke-batch-1", session_id), _request("smoke-batch-2", session_id)])
        summary["batch"] = [{"id": item.request_id, "option_id": item.option_id} for item in batch]
        gates["batch_ok"] = len(batch) == 2 and all(item.option_text in {"approve", "reject"} for item in batch)

        cell = client.write_memory(
            session_id=session_id,
            memory_system="episodic",
            content="The general runtime smoke wrote this episode.",
            summary="general runtime smoke episode",
        )
        summary["explicit_memory_cell"] = cell
        gates["explicit_memory_written"] = bool(cell.get("cell_id"))
        retrieved = client.read_memory(
            session_id=session_id,
            query="general runtime smoke episode",
            memory_system="episodic",
            limit=4,
        )
        summary["memory_read_count"] = len(retrieved)
        gates["memory_read_back"] = len(retrieved) >= 1

        job = client.submit_job(_request("smoke-job-1", session_id))
        finished = client.wait_job(job.job_id, timeout=300.0, poll_interval=0.5)
        summary["async_job"] = {"job_id": finished.job_id, "status": finished.status}
        gates["async_job_completed"] = finished.status == "completed"
        job_result_payload = client.job_result(finished.job_id)
        result_body = job_result_payload.get("result")
        if not isinstance(result_body, dict):
            result_body = job_result_payload
        summary["async_job_result"] = {
            "status": job_result_payload.get("status"),
            "total_rows": result_body.get("total_rows"),
            "row_count": len(result_body.get("rows", [])),
            "first_row_option_id": (
                result_body.get("rows", [{}])[0].get("response", {}).get("predicted_option_id")
                if result_body.get("rows")
                else None
            ),
        }
        gates["async_job_result_present"] = bool(result_body.get("rows"))

        export = client.create_export([finished.job_id])
        export_id = export.get("export", {}).get("export_id") or export.get("export_id")
        package = client.create_package(str(export_id))
        destination = output / f"{export_id}.tar.gz"
        manifest = client.get_package(str(export_id))
        expected_sha256 = (manifest.get("package") or {}).get("sha256") or manifest.get("sha256")
        download = client.download_package(str(export_id), destination, expected_sha256=expected_sha256)
        summary["export"] = {
            "export_id": export_id,
            "package_keys": sorted(package.keys()),
            "download": download,
        }
        gates["export_package_downloaded"] = download["bytes"] > 0 and bool(download["sha256"])
        gates["export_package_sha256_verified"] = download["verified"] is True

        global_store = client.save_global_store()
        gates["global_store_saved"] = global_store.get("status") == "ok"
        gates["global_store_loaded"] = client.load_global_store().get("status") == "ok"

    summary["passes_general_runtime_gate"] = all(gates.values())
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def serve(
    *,
    provider_base_url: str,
    provider_model: str,
    api_key_env: str,
    output_dir: str | Path,
    allow_insecure_http: bool,
    headers: dict[str, str] | None = None,
) -> None:
    """Start the service and keep it up for external SDKs until SIGTERM."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _config(
        provider_base_url=provider_base_url,
        provider_model=provider_model,
        api_key_env=api_key_env,
        output_dir=output / "state",
        allow_insecure_http=allow_insecure_http,
        headers=headers or {},
    )
    runtime = EmbeddedCivilization(config)
    client = runtime.start()
    base_url = client.base_url
    (output / "base_url.txt").write_text(base_url + "\n", encoding="utf-8")
    print(json.dumps({"status": "serving", "base_url": base_url}, ensure_ascii=False), flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.wait(1.0):
        pass
    runtime.shutdown()
    print(json.dumps({"status": "stopped"}, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-base-url", required=True)
    parser.add_argument("--provider-model", required=True)
    parser.add_argument("--api-key-env", default="CIVILIZATION_SMOKE_API_KEY")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra provider header, repeatable (for example a required session header)",
    )
    parser.add_argument("--serve", action="store_true", help="keep serving after the smoke run")
    parser.add_argument("--serve-only", action="store_true", help="skip the smoke run and only serve")
    arguments = parser.parse_args(argv)

    headers: dict[str, str] = {}
    for item in arguments.header:
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            print(f"invalid --header value: {item!r}; expected NAME=VALUE", file=sys.stderr)
            return 2
        headers[name.strip()] = value

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(arguments.output_dir) if arguments.output_dir else DEFAULT_OUTPUT_ROOT / stamp

    if arguments.api_key_env and not os.environ.get(arguments.api_key_env):
        print(f"missing API key environment variable: {arguments.api_key_env}", file=sys.stderr)
        return 2

    if not arguments.serve_only:
        summary = run_smoke(
            provider_base_url=arguments.provider_base_url,
            provider_model=arguments.provider_model,
            api_key_env=arguments.api_key_env,
            output_dir=output_dir,
            allow_insecure_http=arguments.allow_insecure_http,
            headers=headers,
        )
        print(json.dumps({"status": "smoke", "output_dir": str(output_dir), "gates": summary["gates"]}, ensure_ascii=False, indent=2))
        if not summary["passes_general_runtime_gate"]:
            return 1

    if arguments.serve:
        serve(
            provider_base_url=arguments.provider_base_url,
            provider_model=arguments.provider_model,
            api_key_env=arguments.api_key_env,
            output_dir=output_dir,
            allow_insecure_http=arguments.allow_insecure_http,
            headers=headers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
