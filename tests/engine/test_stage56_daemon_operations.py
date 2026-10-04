from __future__ import annotations

import os
from pathlib import Path
import socket

from civilization.engine.stages.stage56_daemon_operations import (
    Stage56DaemonConfig,
    port_available,
    read_pid_status,
    rotate_log_file,
    run_stage56_fake_daemon_smoke,
    validate_stage56_daemon_bundle,
    write_stage56_daemon_bundle,
)


def test_stage56_bundle_contains_daemon_hardening(tmp_path) -> None:
    output = tmp_path / "daemon"
    config = Stage56DaemonConfig(
        output_dir=str(output),
        log_dir=str(output / "logs"),
        pid_file=str(output / "logs" / "daemon.pid"),
        log_file=str(output / "logs" / "daemon.log"),
        model_path="/models/qwen",
    )
    manifest = write_stage56_daemon_bundle(output_dir=output, config=config)
    validation = validate_stage56_daemon_bundle(output_dir=output)
    assert validation["passes_stage_gate"]
    start = Path(manifest["files"]["start_script"])
    stop = Path(manifest["files"]["stop_script"])
    assert os.access(start, os.X_OK)
    assert "nohup" in start.read_text(encoding="utf-8")
    assert "--check-port" in start.read_text(encoding="utf-8")
    assert "--rotate-log" in start.read_text(encoding="utf-8")
    assert "stale_pid" in stop.read_text(encoding="utf-8")


def test_stage56_pid_status_and_log_rotation(tmp_path) -> None:
    pid_file = tmp_path / "stale.pid"
    pid_file.write_text("99999999", encoding="utf-8")
    status = read_pid_status(pid_file)
    assert status["exists"]
    assert status["stale"]
    log_file = tmp_path / "daemon.log"
    log_file.write_text("0123456789abcdef", encoding="utf-8")
    result = rotate_log_file(log_file, max_bytes=4, backup_count=2)
    assert result["rotated"]
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == ""
    assert (tmp_path / "daemon.log.1").read_text(encoding="utf-8") == "0123456789abcdef"


def test_stage56_port_available_detects_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        assert not port_available(host, port)


def test_stage56_fake_daemon_smoke_passes(tmp_path) -> None:
    summary = run_stage56_fake_daemon_smoke(output_dir=tmp_path, port=0)
    assert summary["passes_stage_gate"]
    assert summary["stage_gates"]["bundle_valid"]
    assert summary["stage_gates"]["log_rotated"]
    assert summary["stage_gates"]["stale_pid_detected"]
    assert summary["stage_gates"]["port_occupied_detected"]
    assert (tmp_path / "summary.json").exists()
