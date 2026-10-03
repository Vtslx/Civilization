from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage45_adapter_package import (
    Stage45InferenceResponse,
)
from experiments.civilization_transformer_qwen3.analysis.stage48_production_jsonl_inference import (
    Stage48Limits,
    _control_trace_valid,
    parse_stage48_record,
    run_stage48_jsonl_inference,
)


class FakeRuntime:
    def predict(self, request):
        if request.text == "runtime failure":
            raise RuntimeError("synthetic runtime failure")
        return Stage45InferenceResponse(
            status="ok",
            task_name=request.task_name,
            seed=request.seed,
            predicted_option_id=0,
            scores={"projected_delta": [1.0, 0.0]},
            trace={"control_mode": request.control_mode},
        )


def _record(**overrides):
    payload = {
        "id": "request-1",
        "text": "Evaluate the operation.",
        "memory_items": ["memory evidence"],
        "rule_items": ["rule evidence"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "operation_decision",
        "seed": 202,
    }
    payload.update(overrides)
    return payload


def test_stage48_parser_expands_valid_controls() -> None:
    request_id, request, controls = parse_stage48_record(
        _record(controls=["full", "no_memory_path"]),
        limits=Stage48Limits(),
        default_controls=("full",),
    )
    assert request_id == "request-1"
    assert request.control_mode == "full"
    assert controls == ("full", "no_memory_path")


def test_stage48_parser_rejects_unsupported_control() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_stage48_record(
            _record(controls=["unknown"]),
            limits=Stage48Limits(),
            default_controls=("full",),
        )


def test_stage48_jsonl_isolates_failures_and_writes_atomically(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(_record(controls=["full", "no_rule_path"])),
                "{invalid json",
                json.dumps(_record(id="request-3", text="runtime failure")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "responses.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run_stage48_jsonl_inference(
        runtime=FakeRuntime(),
        input_path=source,
        output_path=output,
        summary_path=summary_path,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["input_count"] == 3
    assert summary["output_count"] == 4
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 2
    assert [row.get("control_mode") for row in rows[:2]] == ["full", "no_rule_path"]
    assert not output.with_suffix(".jsonl.tmp").exists()
    assert summary_path.exists()


def test_stage48_enforces_context_limits() -> None:
    with pytest.raises(ValueError, match="context item count"):
        parse_stage48_record(
            _record(memory_items=["x", "y"]),
            limits=Stage48Limits(max_context_items=1),
            default_controls=("full",),
        )


def test_stage48_control_trace_audit_checks_zeroed_path() -> None:
    response = Stage45InferenceResponse(
        status="ok",
        task_name="task",
        seed=202,
        predicted_option_id=0,
        scores={},
        trace={"traces": {"16": {"memory_delta_norm": 0.0, "rule_delta_norm": 1.0, "state_delta_norm": 1.0, "delta_norm": 1.0}}},
    )
    assert _control_trace_valid(response, "no_memory_path")
    assert not _control_trace_valid(response, "no_rule_path")
