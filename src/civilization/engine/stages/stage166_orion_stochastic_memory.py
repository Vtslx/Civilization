"""Experimental stochastic token reservation with output reconciliation.

This module deliberately has no model, storage, or network dependency. A local
token runtime can supply logits for candidate output positions; this component
reserves one reproducible token, reconciles it against emitted output, and only
then reserves a provenance-bound memory token. The emitted output is always the
source of truth, and a one-token memory draft is never persisted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage166_orion_stochastic_memory")
ALGORITHM_VERSION = "sha256-counter-categorical-v1"
_SEED_LIMIT = 1 << 128


@dataclass(frozen=True)
class TokenCandidatePrediction:
    position: int
    token_id: int
    probability: float


@dataclass(frozen=True)
class StochasticOutputReservation:
    transaction_id: str
    random_seed: int
    context_fingerprint: str
    candidate_predictions: tuple[TokenCandidatePrediction, ...]
    selected_position: int
    reserved_token_id: int
    temperature: float
    top_k: int
    algorithm_version: str = ALGORITHM_VERSION


@dataclass(frozen=True)
class OutputReconciliation:
    transaction_id: str
    random_seed: int
    context_fingerprint: str
    status: Literal["pending", "accepted", "corrected", "discarded"]
    selected_position: int
    reserved_token_id: int
    observed_token_id: int | None
    rollback_from_position: int | None
    merged_output_tokens: tuple[int, ...]
    output_sha256: str
    output_complete: bool


@dataclass(frozen=True)
class StochasticMemoryReservation:
    transaction_id: str
    random_seed: int
    context_fingerprint: str
    reconciled_output_sha256: str
    output_token_count: int
    position: int
    reserved_token_id: int
    probability: float
    source_cell_ids: tuple[str, ...]
    status: Literal["reserved"] = "reserved"
    persistent_write_performed: bool = False


def _validate_transaction(transaction_id: str, random_seed: int, context_fingerprint: str) -> None:
    if not isinstance(transaction_id, str) or not transaction_id.strip() or len(transaction_id) > 128:
        raise ValueError("transaction_id must contain 1 to 128 non-whitespace characters")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not 0 <= random_seed < _SEED_LIMIT:
        raise ValueError("random_seed must be an unsigned 128-bit integer")
    if (
        not isinstance(context_fingerprint, str)
        or len(context_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in context_fingerprint)
    ):
        raise ValueError("context_fingerprint must be a lowercase SHA-256 hex digest")


def _validate_sampling(temperature: float, top_k: int) -> tuple[float, int]:
    try:
        normalized_temperature = float(temperature)
    except (TypeError, ValueError) as error:
        raise ValueError("temperature must be finite and greater than zero") from error
    if not math.isfinite(normalized_temperature) or normalized_temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return normalized_temperature, top_k


def _normalize_logits(logits: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in logits)
    except (TypeError, ValueError) as error:
        raise ValueError("logits must be a non-empty sequence of finite numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logits must be a non-empty sequence of finite numbers")
    return values


def _uniform_draw(random_seed: int, transaction_id: str, stream: str, counter: int = 0) -> float:
    """Return a stable [0, 1) draw without depending on process-global RNG state."""

    material = json.dumps(
        [ALGORITHM_VERSION, random_seed, transaction_id, stream, counter],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return draw / float(1 << 64)


def _sample_token(
    logits: Sequence[float],
    *,
    temperature: float,
    top_k: int,
    draw: float,
) -> tuple[int, float]:
    values = _normalize_logits(logits)
    candidates = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))[: min(top_k, len(values))]
    maximum = candidates[0][1]
    weighted = [(token_id, math.exp((value - maximum) / temperature)) for token_id, value in candidates]
    total = sum(weight for _, weight in weighted)
    threshold = draw * total
    cumulative = 0.0
    for token_id, weight in weighted:
        cumulative += weight
        if threshold < cumulative:
            return token_id, weight / total
    token_id, weight = weighted[-1]
    return token_id, weight / total


def _validate_token_ids(tokens: Sequence[int], *, label: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} must contain only non-negative integer token IDs")
        normalized.append(token)
    return tuple(normalized)


def _token_digest(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def prepare_stochastic_output_reservation(
    *,
    transaction_id: str,
    random_seed: int,
    context_fingerprint: str,
    candidate_positions: Sequence[int],
    logits_by_position: Mapping[int, Sequence[float]],
    temperature: float = 1.0,
    top_k: int = 32,
) -> StochasticOutputReservation:
    """Predict tokens at multiple draft offsets and reserve one seeded candidate."""

    _validate_transaction(transaction_id, random_seed, context_fingerprint)
    normalized_temperature, normalized_top_k = _validate_sampling(temperature, top_k)
    positions = tuple(candidate_positions)
    if len(positions) < 2:
        raise ValueError("at least two candidate_positions are required")
    if any(isinstance(position, bool) or not isinstance(position, int) or position < 0 for position in positions):
        raise ValueError("candidate_positions must contain non-negative integer offsets")
    if len(set(positions)) != len(positions):
        raise ValueError("candidate_positions must be unique")
    if set(logits_by_position) != set(positions):
        raise ValueError("logits_by_position must contain exactly the candidate positions")

    normalized_positions = tuple(sorted(positions))
    normalized_logits = {position: _normalize_logits(logits_by_position[position]) for position in normalized_positions}
    if len({len(values) for values in normalized_logits.values()}) != 1:
        raise ValueError("all candidate positions must use the same vocabulary size")

    predictions: list[TokenCandidatePrediction] = []
    for position in normalized_positions:
        token_id, probability = _sample_token(
            normalized_logits[position],
            temperature=normalized_temperature,
            top_k=normalized_top_k,
            draw=_uniform_draw(random_seed, transaction_id, f"output-candidate:{position}"),
        )
        predictions.append(TokenCandidatePrediction(position, token_id, probability))

    position_draw = _uniform_draw(random_seed, transaction_id, "output-position")
    selected_index = min(int(position_draw * len(predictions)), len(predictions) - 1)
    selected = predictions[selected_index]
    return StochasticOutputReservation(
        transaction_id=transaction_id,
        random_seed=random_seed,
        context_fingerprint=context_fingerprint,
        candidate_predictions=tuple(predictions),
        selected_position=selected.position,
        reserved_token_id=selected.token_id,
        temperature=normalized_temperature,
        top_k=normalized_top_k,
    )


def reconcile_emitted_output(
    reservation: StochasticOutputReservation,
    emitted_output_tokens: Sequence[int],
    *,
    output_complete: bool,
) -> OutputReconciliation:
    """Accept an exact match or roll back to the emitted output as authority."""

    if not isinstance(reservation, StochasticOutputReservation):
        raise TypeError("reservation must be a StochasticOutputReservation")
    if not isinstance(output_complete, bool):
        raise ValueError("output_complete must be a boolean")
    emitted = _validate_token_ids(emitted_output_tokens, label="emitted_output_tokens")
    position = reservation.selected_position
    observed = emitted[position] if position < len(emitted) else None

    if observed is None and not output_complete:
        status: Literal["pending", "accepted", "corrected", "discarded"] = "pending"
        rollback_position = None
    elif observed == reservation.reserved_token_id:
        status = "accepted"
        rollback_position = None
    elif observed is None:
        status = "discarded"
        rollback_position = position
    else:
        status = "corrected"
        rollback_position = position

    # Never merge an unverified draft token into the returned output. The
    # emitted stream is authoritative in both the accept and rollback cases.
    merged = emitted
    return OutputReconciliation(
        transaction_id=reservation.transaction_id,
        random_seed=reservation.random_seed,
        context_fingerprint=reservation.context_fingerprint,
        status=status,
        selected_position=position,
        reserved_token_id=reservation.reserved_token_id,
        observed_token_id=observed,
        rollback_from_position=rollback_position,
        merged_output_tokens=merged,
        output_sha256=_token_digest(merged),
        output_complete=output_complete,
    )


def reserve_next_memory_token(
    reconciliation: OutputReconciliation,
    memory_logits: Sequence[float],
    *,
    conditioned_on_output_sha256: str,
    source_cell_ids: Sequence[str],
    temperature: float = 1.0,
    top_k: int = 32,
) -> StochasticMemoryReservation:
    """Reserve one memory token from the reconciled output state, without writing it.

    The caller supplies the digest of the exact output state used to compute
    ``memory_logits``. This prevents accidentally carrying a reservation across
    an output correction or stream change.
    """

    if not isinstance(reconciliation, OutputReconciliation):
        raise TypeError("reconciliation must be an OutputReconciliation")
    if not reconciliation.output_complete or reconciliation.status == "pending":
        raise ValueError("memory reservation requires complete, reconciled output")
    if conditioned_on_output_sha256 != reconciliation.output_sha256:
        raise ValueError("memory logits must be conditioned on the reconciled output digest")
    _validate_transaction(
        reconciliation.transaction_id,
        reconciliation.random_seed,
        reconciliation.context_fingerprint,
    )
    normalized_temperature, normalized_top_k = _validate_sampling(temperature, top_k)
    sources = tuple(source_cell_ids)
    if not sources or any(not isinstance(source, str) or not source.strip() for source in sources):
        raise ValueError("source_cell_ids must contain at least one non-empty source ID")
    if len(set(sources)) != len(sources):
        raise ValueError("source_cell_ids must be unique")
    logits = _normalize_logits(memory_logits)
    stream = f"memory-token:{reconciliation.output_sha256}"
    token_id, probability = _sample_token(
        logits,
        temperature=normalized_temperature,
        top_k=normalized_top_k,
        draw=_uniform_draw(reconciliation.random_seed, reconciliation.transaction_id, stream),
    )
    return StochasticMemoryReservation(
        transaction_id=reconciliation.transaction_id,
        random_seed=reconciliation.random_seed,
        context_fingerprint=reconciliation.context_fingerprint,
        reconciled_output_sha256=reconciliation.output_sha256,
        output_token_count=len(reconciliation.merged_output_tokens),
        position=0,
        reserved_token_id=token_id,
        probability=probability,
        source_cell_ids=sources,
    )


def run_stage166_orion_stochastic_memory_smoke(
    *, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    """Run a deterministic mismatch/rollback/memory-reservation demonstration."""

    context_fingerprint = hashlib.sha256(b"stage166 deterministic synthetic context").hexdigest()
    reservation = prepare_stochastic_output_reservation(
        transaction_id="stage166-smoke-001",
        random_seed=1662026,
        context_fingerprint=context_fingerprint,
        candidate_positions=(1, 3, 5),
        logits_by_position={
            1: (0.1, 2.0, -0.5, 0.7),
            3: (1.4, 0.3, 0.0, 2.1),
            5: (-0.2, 1.1, 1.8, 0.4),
        },
        temperature=0.8,
        top_k=3,
    )
    emitted = [0, 0, 0, 0, 0, 0]
    emitted[reservation.selected_position] = (reservation.reserved_token_id + 1) % 4
    reconciliation = reconcile_emitted_output(reservation, emitted, output_complete=True)
    memory_reservation = reserve_next_memory_token(
        reconciliation,
        (0.0, 0.8, 2.0, 0.2),
        conditioned_on_output_sha256=reconciliation.output_sha256,
        source_cell_ids=("episodic-000001", "episodic-000002"),
        temperature=0.9,
        top_k=3,
    )
    summary: dict[str, Any] = {
        "stage": "stage166_orion_stochastic_memory",
        "algorithm_version": ALGORITHM_VERSION,
        "candidate_predictions": [asdict(item) for item in reservation.candidate_predictions],
        "selected_position": reservation.selected_position,
        "reserved_output_token_id": reservation.reserved_token_id,
        "output_reconciliation": {
            "status": reconciliation.status,
            "observed_token_id": reconciliation.observed_token_id,
            "rollback_from_position": reconciliation.rollback_from_position,
            "merged_output_tokens": list(reconciliation.merged_output_tokens),
            "output_sha256": reconciliation.output_sha256,
        },
        "memory_reservation": asdict(memory_reservation),
        "persistent_memory_writes": 0,
        "stage_gates": {
            "multiple_positions_predicted": len(reservation.candidate_predictions) >= 2,
            "reserved_token_is_selected_candidate": reservation.reserved_token_id
            == next(
                item.token_id
                for item in reservation.candidate_predictions
                if item.position == reservation.selected_position
            ),
            "emitted_output_wins_on_mismatch": reconciliation.status == "corrected"
            and reconciliation.merged_output_tokens == tuple(emitted)
            and reconciliation.observed_token_id != reservation.reserved_token_id,
            "memory_reserved_after_output_reconciliation": memory_reservation.reconciled_output_sha256
            == reconciliation.output_sha256,
            "memory_reservation_has_provenance": bool(memory_reservation.source_cell_ids),
            "draft_not_persisted": memory_reservation.persistent_write_performed is False,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
