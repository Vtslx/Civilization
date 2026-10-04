"""Stable public data models for the Civilization v1 SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4


class MemorySystem(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True)
class CivilizationRequest:
    """One production decision request.

    The SDK deliberately does not expose online ablation controls. Every
    serialized request uses the validated production ``full`` path.
    """

    text: str
    answer_options: Sequence[str]
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str = "default"
    memory_items: Sequence[str] = ()
    rule_items: Sequence[str] = ()
    state_values: Sequence[float] = (0.5, 0.5, 0.5)
    task_name: str = "custom"
    seed: int = 202
    read_memory: bool = True
    write_memory: bool = True
    include_global_memory: bool = False
    allow_resolved_global_memory: bool = False
    readouts: Sequence[str] = ("api_choice",)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        options = _strings(self.answer_options, "answer_options")
        if len(options) < 2:
            raise ValueError("answer_options must contain at least two options")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.session_id):
            raise ValueError("session_id contains unsupported characters")
        if not isinstance(self.task_name, str) or not self.task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        state = tuple(self.state_values)
        if len(state) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in state
        ):
            raise ValueError("state_values must contain exactly three numbers")
        if any(
            not isinstance(value, bool)
            for value in (
                self.read_memory,
                self.write_memory,
                self.include_global_memory,
                self.allow_resolved_global_memory,
            )
        ):
            raise ValueError("memory controls must be booleans")
        object.__setattr__(self, "answer_options", options)
        object.__setattr__(self, "memory_items", _strings(self.memory_items, "memory_items"))
        object.__setattr__(self, "rule_items", _strings(self.rule_items, "rule_items"))
        object.__setattr__(self, "state_values", tuple(float(value) for value in state))
        object.__setattr__(self, "readouts", _strings(self.readouts, "readouts"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "text": self.text,
            "memory_items": list(_strings(self.memory_items, "memory_items")),
            "rule_items": list(_strings(self.rule_items, "rule_items")),
            "state_values": [float(value) for value in self.state_values],
            "answer_options": list(_strings(self.answer_options, "answer_options")),
            "task_name": self.task_name,
            "seed": int(self.seed),
            "controls": ["full"],
            "readouts": list(_strings(self.readouts, "readouts")),
            "orion_memory": {
                "session_id": self.session_id,
                "read": self.read_memory,
                "write": self.write_memory,
                "include_global": self.include_global_memory,
                "allow_resolved_global": self.allow_resolved_global_memory,
            },
        }


@dataclass(frozen=True)
class Prediction:
    request_id: str
    option_id: int
    option_text: str
    scores: Mapping[str, tuple[float, ...]]
    trace: Mapping[str, Any]
    memory_trace: Mapping[str, Any]
    raw: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_row(cls, request: CivilizationRequest, row: Mapping[str, Any]) -> "Prediction":
        response = row.get("response")
        if row.get("status") != "ok" or not isinstance(response, Mapping):
            error = row.get("error", "inference returned a non-ok row")
            raise ValueError(str(error))
        option_id = response.get("predicted_option_id")
        options = tuple(request.answer_options)
        if isinstance(option_id, bool) or not isinstance(option_id, int) or not 0 <= option_id < len(options):
            raise ValueError("inference response contains an invalid predicted_option_id")
        raw_scores = response.get("scores", {})
        if not isinstance(raw_scores, Mapping):
            raise ValueError("inference response scores must be an object")
        scores = {
            str(name): tuple(float(value) for value in values)
            for name, values in raw_scores.items()
            if isinstance(values, (list, tuple))
        }
        trace = response.get("trace", {})
        memory_trace = row.get("orion_memory", {})
        return cls(
            request_id=str(row.get("id", request.request_id)),
            option_id=option_id,
            option_text=options[option_id],
            scores=MappingProxyType(scores),
            trace=MappingProxyType(dict(trace)) if isinstance(trace, Mapping) else MappingProxyType({}),
            memory_trace=(
                MappingProxyType(dict(memory_trace))
                if isinstance(memory_trace, Mapping)
                else MappingProxyType({})
            ),
            raw=MappingProxyType(dict(row)),
        )


@dataclass(frozen=True)
class Job:
    job_id: str
    status: str
    result: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    raw: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Job":
        job = payload.get("job", payload)
        if not isinstance(job, Mapping) or not job.get("job_id"):
            raise ValueError("response does not contain a job")
        result = job.get("result")
        error = job.get("error")
        return cls(
            job_id=str(job["job_id"]),
            status=str(job.get("status", "unknown")),
            result=MappingProxyType(dict(result)) if isinstance(result, Mapping) else None,
            error=MappingProxyType(dict(error)) if isinstance(error, Mapping) else None,
            raw=MappingProxyType(dict(job)),
        )
