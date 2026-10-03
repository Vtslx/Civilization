from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from civilization_v1 import CivilizationRequest, EmbeddedCivilization, EmbeddedConfig, MemorySystem


DEFAULT_OUTPUT_ROOT = Path(
    "experiments/civilization_transformer_qwen3/artifacts/v1_sdk_production_smoke"
)


def _request(request_id: str, session_id: str) -> CivilizationRequest:
    return CivilizationRequest(
        request_id=request_id,
        text="Choose the deployment action supported by the verified evidence.",
        answer_options=("approve", "reject"),
        session_id=session_id,
        task_name="sdk_production_smoke",
        memory_items=("The deployment signature and health checks are valid.",),
        rule_items=("Reject only when a critical verification is unresolved.",),
        state_values=(0.9, 0.1, 0.8),
    )


def run_smoke(
    *,
    provider_base_url: str,
    provider_model: str,
    api_key_env: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    session_id = "sdk-production-smoke"
    config = EmbeddedConfig(
        provider_base_url=provider_base_url,
        provider_model=provider_model,
        provider_api_key_env=api_key_env,
        bearer_token_env=None,
        state_dir=str(output / "state"),
    )
    with EmbeddedCivilization(config) as runtime:
        client = runtime.start()
        explicit = client.write_memory(
            session_id=session_id,
            memory_system=MemorySystem.EPISODIC,
            content="A prior verified deployment completed successfully.",
            summary="prior verified deployment",
            metadata={"source": "sdk-production-smoke"},
        )
        first = client.predict(_request("sdk-production-sync-1", session_id))
        second = client.predict(_request("sdk-production-sync-2", session_id))
        submitted = client.submit_job(_request("sdk-production-async", session_id))
        completed = client.wait_job(submitted.job_id, timeout=180.0, poll_interval=0.2)
        status = client.memory_status()
        saved = client.save_global_store()
        loaded = client.load_global_store()
        controls = [
            first.raw.get("control_mode"),
            second.raw.get("control_mode"),
        ]
        runtime_names = [first.trace.get("runtime"), second.trace.get("runtime")]
        gates = {
            "ready": client.ready().get("ready") is True,
            "sync_predictions_ok": first.option_id in {0, 1} and second.option_id in {0, 1},
            "production_full_only": controls == ["full", "full"],
            "external_runtime_used": all(name == "openai_compatible_chat_completions" for name in runtime_names),
            "adapter_claim_is_false": all(trace.get("adapter_execution") is False for trace in (first.trace, second.trace)),
            "memory_retrieved": second.memory_trace.get("injected_item_count", 0) > 0,
            "async_job_completed": completed.status == "completed",
            "explicit_memory_written": explicit.get("memory_system") == "episodic",
            "global_store_saved": saved.get("status") == "ok",
            "global_store_loaded": loaded.get("status") == "ok",
        }
        summary = {
            "sdk_version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": {"base_url": provider_base_url, "model": provider_model},
            "config": {
                "embedded": asdict(config),
                "api_key_present_in_summary": False,
            },
            "sync": {
                "selected_option_ids": [first.option_id, second.option_id],
                "controls": controls,
                "runtime_names": runtime_names,
                "memory_injected_item_counts": [
                    first.memory_trace.get("injected_item_count", 0),
                    second.memory_trace.get("injected_item_count", 0),
                ],
            },
            "async": {"job_id": completed.job_id, "status": completed.status},
            "memory": {
                "session_count": status.get("session_count"),
                "explicit_cell_id": explicit.get("cell_id"),
            },
            "gates": gates,
            "passes_sdk_gate": all(gates.values()),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Civilization v1 SDK production smoke")
    parser.add_argument("--provider-base-url", required=True)
    parser.add_argument("--provider-model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output = args.output_dir or str(
        DEFAULT_OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    summary = run_smoke(
        provider_base_url=args.provider_base_url,
        provider_model=args.provider_model,
        api_key_env=args.api_key_env,
        output_dir=output,
    )
    print(json.dumps({"output_dir": output, "passes_sdk_gate": summary["passes_sdk_gate"]}))


if __name__ == "__main__":
    main()
