from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import RosetteContextAssembler, RosetteContextPacket, _create_lagoon_schema
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage142_rosette_packet_snapshot")
ROSETTE_PACKET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RosettePacketSnapshotManifest:
    schema_version: int
    packet_id: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RosettePacketSnapshot:
    """Persists canonical Rosette packets with an atomic SHA-256 envelope."""

    def save(self, packet: RosetteContextPacket, path: str | Path) -> RosettePacketSnapshotManifest:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        packet_bytes = self.canonical_packet_bytes(packet)
        digest = hashlib.sha256(packet_bytes).hexdigest()
        manifest = RosettePacketSnapshotManifest(ROSETTE_PACKET_SCHEMA_VERSION, packet.packet_id, digest)
        envelope = {"manifest": manifest.to_dict(), "packet": packet.to_dict()}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
        return manifest

    @staticmethod
    def canonical_packet_bytes(packet: RosetteContextPacket) -> bytes:
        return json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run_stage142_rosette_packet_snapshot_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    packet = RosetteContextAssembler().assemble(store, binder, "laboratory collect")
    snapshot = RosettePacketSnapshot()
    first_path, second_path = output / "packet.json", output / "packet-repeat.json"
    first = snapshot.save(packet, first_path)
    second = snapshot.save(packet, second_path)
    envelope = json.loads(first_path.read_text(encoding="utf-8"))
    summary = {
        "stage": "stage142_rosette_packet_snapshot",
        "manifest": first.to_dict(),
        "stage_gates": {
            "snapshot_written": first_path.exists() and second_path.exists(),
            "schema_version_frozen": first.schema_version == ROSETTE_PACKET_SCHEMA_VERSION == 1,
            "packet_id_preserved": first.packet_id == packet.packet_id and envelope["manifest"]["packet_id"] == packet.packet_id,
            "hash_is_reproducible": first.sha256 == second.sha256 and first_path.read_bytes() == second_path.read_bytes(),
            "canonical_hash_matches": first.sha256 == hashlib.sha256(snapshot.canonical_packet_bytes(packet)).hexdigest(),
            "temporary_file_removed": not first_path.with_name(f".{first_path.name}.tmp").exists(),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output, summary=summary)
    return summary
