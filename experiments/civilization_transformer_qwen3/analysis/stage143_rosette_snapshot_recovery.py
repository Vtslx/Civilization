from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import RosetteContextAssembler, RosetteContextPacket, _create_lagoon_schema
from .stage141_rosette_packet_validator import RosettePacketValidator
from .stage142_rosette_packet_snapshot import ROSETTE_PACKET_SCHEMA_VERSION, RosettePacketSnapshot
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage143_rosette_snapshot_recovery")


@dataclass(frozen=True)
class RosettePacketRecoveryResult:
    recovered: bool
    reason: str
    packet: RosetteContextPacket | None
    validation_errors: tuple[str, ...] = ()


class RosettePacketRecovery:
    """Recovers checksummed Rosette packets and revalidates them against live memory state."""

    def recover(self, path: str | Path, store: OrionMemoryStore, binder: TrifidEpisodeBinder) -> RosettePacketRecoveryResult:
        try:
            envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return RosettePacketRecoveryResult(False, "invalid_json", None)
        try:
            manifest = envelope["manifest"]
            packet_payload = envelope["packet"]
            if int(manifest["schema_version"]) != ROSETTE_PACKET_SCHEMA_VERSION:
                return RosettePacketRecoveryResult(False, "unsupported_schema_version", None)
            packet = RosetteContextPacket.from_dict(packet_payload)
            if manifest["packet_id"] != packet.packet_id:
                return RosettePacketRecoveryResult(False, "packet_id_mismatch", None)
            digest = hashlib.sha256(RosettePacketSnapshot.canonical_packet_bytes(packet)).hexdigest()
            if manifest["sha256"] != digest:
                return RosettePacketRecoveryResult(False, "checksum_mismatch", None)
        except (KeyError, TypeError, ValueError):
            return RosettePacketRecoveryResult(False, "invalid_schema", None)
        validation = RosettePacketValidator().validate(store, binder, packet)
        if not validation.valid:
            return RosettePacketRecoveryResult(False, "packet_validation_failed", None, validation.errors)
        return RosettePacketRecoveryResult(True, "ok", packet)


def run_stage143_rosette_snapshot_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    packet = RosetteContextAssembler().assemble(store, binder, "laboratory collect")
    snapshot = RosettePacketSnapshot()
    valid_path = output / "valid-packet.json"
    snapshot.save(packet, valid_path)
    recovery = RosettePacketRecovery()
    cell_count, link_count = len(store.cells), len(store.links)
    valid = recovery.recover(valid_path, store, binder)

    tampered_path = output / "tampered-packet.json"
    tampered_envelope = json.loads(valid_path.read_text(encoding="utf-8"))
    tampered_envelope["packet"]["query"] = "tampered query"
    tampered_path.write_text(json.dumps(tampered_envelope, ensure_ascii=False), encoding="utf-8")
    tampered = recovery.recover(tampered_path, store, binder)

    group = packet.groups[0]
    dangling_group = replace(group, factual_cell_ids=(*group.factual_cell_ids, "episodic-999999"))
    dangling_groups = (dangling_group,)
    dangling_packet = RosetteContextPacket(RosetteContextAssembler._packet_id(packet.query, list(dangling_groups)), packet.query, dangling_groups)
    dangling_path = output / "dangling-packet.json"
    snapshot.save(dangling_packet, dangling_path)
    dangling = recovery.recover(dangling_path, store, binder)

    invalid_path = output / "invalid-packet.json"
    invalid_path.write_text("{not-json", encoding="utf-8")
    invalid = recovery.recover(invalid_path, store, binder)
    summary = {
        "stage": "stage143_rosette_snapshot_recovery",
        "stage_gates": {
            "valid_snapshot_recovered": valid.recovered and valid.packet == packet and valid.reason == "ok",
            "tampered_snapshot_rejected": not tampered.recovered and tampered.reason == "checksum_mismatch",
            "dangling_reference_rejected": not dangling.recovered and dangling.reason == "packet_validation_failed" and "missing_or_inactive_reference" in dangling.validation_errors,
            "invalid_json_rejected": not invalid.recovered and invalid.reason == "invalid_json",
            "recovery_is_read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "recovered_packet_revalidated": any(event.details.get("rosette_packet_validation") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output, summary=summary)
    return summary
