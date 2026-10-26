from __future__ import annotations

import hashlib

import pytest

from civilization.engine.stages.stage166_orion_stochastic_memory import (
    prepare_stochastic_output_reservation,
    reconcile_emitted_output,
    reserve_next_memory_token,
    run_stage166_orion_stochastic_memory_smoke,
)


def _context_fingerprint() -> str:
    return hashlib.sha256(b"controlled test context").hexdigest()


def _reservation(*, seed: int = 1234):
    return prepare_stochastic_output_reservation(
        transaction_id="test-transaction-1",
        random_seed=seed,
        context_fingerprint=_context_fingerprint(),
        candidate_positions=(2, 4, 7),
        logits_by_position={
            2: (0.0, 0.2, 3.0, 0.1),
            4: (0.1, 2.0, 0.4, 0.0),
            7: (1.0, 0.1, 0.0, 2.1),
        },
        temperature=0.8,
        top_k=3,
    )


def test_stage166_predicts_multiple_positions_and_replays_seeded_state() -> None:
    first = _reservation()
    replay = _reservation()

    assert first == replay
    assert [item.position for item in first.candidate_predictions] == [2, 4, 7]
    selected = next(item for item in first.candidate_predictions if item.position == first.selected_position)
    assert first.reserved_token_id == selected.token_id
    assert selected.probability > 0


def test_stage166_different_seed_changes_the_reservation_stream() -> None:
    assert _reservation(seed=1234) != _reservation(seed=5678)


def test_stage166_exact_emitted_token_accepts_reservation_without_rewriting_output() -> None:
    reservation = _reservation()
    emitted = [9, 8, 7, 6, 5, 4, 3, 2]
    emitted[reservation.selected_position] = reservation.reserved_token_id

    result = reconcile_emitted_output(reservation, emitted, output_complete=True)

    assert result.status == "accepted"
    assert result.rollback_from_position is None
    assert result.merged_output_tokens == tuple(emitted)


def test_stage166_mismatch_rolls_back_to_authoritative_emitted_token() -> None:
    reservation = _reservation()
    emitted = [9, 8, 7, 6, 5, 4, 3, 2]
    emitted[reservation.selected_position] = (reservation.reserved_token_id + 1) % 10

    result = reconcile_emitted_output(reservation, emitted, output_complete=True)

    assert result.status == "corrected"
    assert result.observed_token_id == emitted[reservation.selected_position]
    assert result.rollback_from_position == reservation.selected_position
    assert result.merged_output_tokens == tuple(emitted)


def test_stage166_incomplete_output_waits_for_reserved_position() -> None:
    reservation = _reservation()

    result = reconcile_emitted_output(reservation, (), output_complete=False)

    assert result.status == "pending"
    assert result.merged_output_tokens == ()
    with pytest.raises(ValueError, match="complete, reconciled output"):
        reserve_next_memory_token(
            result,
            (0.0, 1.0),
            conditioned_on_output_sha256=result.output_sha256,
            source_cell_ids=("episodic-1",),
        )


def test_stage166_final_output_shorter_than_position_discards_draft() -> None:
    reservation = _reservation()
    emitted = [3]

    result = reconcile_emitted_output(reservation, emitted, output_complete=True)

    assert result.status == "discarded"
    assert result.rollback_from_position == reservation.selected_position
    assert result.merged_output_tokens == tuple(emitted)


def test_stage166_memory_token_is_bound_to_reconciled_output_and_provenance() -> None:
    reservation = _reservation()
    emitted = [0] * 8
    emitted[reservation.selected_position] = (reservation.reserved_token_id + 1) % 4
    reconciled = reconcile_emitted_output(reservation, emitted, output_complete=True)

    first = reserve_next_memory_token(
        reconciled,
        (0.0, 1.0, 2.0, -1.0),
        conditioned_on_output_sha256=reconciled.output_sha256,
        source_cell_ids=("episodic-1", "episodic-2"),
        temperature=0.7,
        top_k=3,
    )
    replay = reserve_next_memory_token(
        reconciled,
        (0.0, 1.0, 2.0, -1.0),
        conditioned_on_output_sha256=reconciled.output_sha256,
        source_cell_ids=("episodic-1", "episodic-2"),
        temperature=0.7,
        top_k=3,
    )

    assert first == replay
    assert first.reconciled_output_sha256 == reconciled.output_sha256
    assert first.output_token_count == len(emitted)
    assert first.source_cell_ids == ("episodic-1", "episodic-2")
    assert first.persistent_write_performed is False


def test_stage166_memory_logits_must_match_the_reconciled_output_digest() -> None:
    reservation = _reservation()
    emitted = [0] * 8
    emitted[reservation.selected_position] = reservation.reserved_token_id
    reconciled = reconcile_emitted_output(reservation, emitted, output_complete=True)

    with pytest.raises(ValueError, match="conditioned on the reconciled output digest"):
        reserve_next_memory_token(
            reconciled,
            (0.0, 1.0),
            conditioned_on_output_sha256="0" * 64,
            source_cell_ids=("episodic-1",),
        )


def test_stage166_memory_reservation_requires_source_provenance() -> None:
    reservation = _reservation()
    emitted = [0] * 8
    emitted[reservation.selected_position] = reservation.reserved_token_id
    reconciled = reconcile_emitted_output(reservation, emitted, output_complete=True)

    with pytest.raises(ValueError, match="source_cell_ids"):
        reserve_next_memory_token(
            reconciled,
            (0.0, 1.0),
            conditioned_on_output_sha256=reconciled.output_sha256,
            source_cell_ids=(),
        )
    with pytest.raises(ValueError, match="unique"):
        reserve_next_memory_token(
            reconciled,
            (0.0, 1.0),
            conditioned_on_output_sha256=reconciled.output_sha256,
            source_cell_ids=("episodic-1", "episodic-1"),
        )


@pytest.mark.parametrize(
    ("positions", "logits", "message"),
    [
        ((1,), {1: (0.0, 1.0)}, "at least two"),
        ((1, 1), {1: (0.0, 1.0)}, "unique"),
        ((-1, 1), {-1: (0.0, 1.0), 1: (0.0, 1.0)}, "non-negative"),
        ((1, 2), {1: (0.0, 1.0)}, "exactly the candidate positions"),
        ((1, 2), {1: (0.0, 1.0), 2: (0.0, float("nan"))}, "finite"),
        ((1, 2), {1: (0.0, 1.0), 2: (0.0, 1.0, 2.0)}, "same vocabulary size"),
    ],
)
def test_stage166_rejects_invalid_candidate_inputs(positions, logits, message) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_stochastic_output_reservation(
            transaction_id="test-transaction-1",
            random_seed=1,
            context_fingerprint=_context_fingerprint(),
            candidate_positions=positions,
            logits_by_position=logits,
        )


def test_stage166_smoke_artifact_is_deterministic_and_contains_no_memory_write(tmp_path) -> None:
    first = run_stage166_orion_stochastic_memory_smoke(output_dir=tmp_path / "first")
    second = run_stage166_orion_stochastic_memory_smoke(output_dir=tmp_path / "second")

    assert first == second
    assert first["passes_stage_gate"] is True
    assert first["persistent_memory_writes"] == 0
    assert first["memory_reservation"]["persistent_write_performed"] is False
    assert (tmp_path / "first" / "summary.json").read_bytes() == (
        tmp_path / "second" / "summary.json"
    ).read_bytes()
