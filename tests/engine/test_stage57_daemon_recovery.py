from __future__ import annotations

import socket

from civilization.engine.stages.stage57_daemon_recovery import (
    Stage57RecoveryConfig,
    _make_daemon_config,
)
from civilization.engine.stages.stage56_daemon_operations import (
    port_available,
    read_pid_status,
    rotate_log_file,
    validate_stage56_daemon_bundle,
    write_stage56_daemon_bundle,
)


def test_stage57_builds_stage56_daemon_config(tmp_path) -> None:
    config = Stage57RecoveryConfig(
        output_dir=str(tmp_path / "recovery"),
        daemon_output_dir=str(tmp_path / "daemon"),
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "logs" / "daemon.pid"),
        log_file=str(tmp_path / "logs" / "daemon.log"),
        model_path="/models/qwen",
        max_log_bytes=16,
        log_backup_count=2,
    )
    daemon = _make_daemon_config(config)
    assert daemon.output_dir == str(tmp_path / "daemon")
    assert daemon.pid_file.endswith("daemon.pid")
    assert daemon.log_file.endswith("daemon.log")
    assert daemon.model_path == "/models/qwen"
    assert daemon.max_log_bytes == 16


def test_stage57_recovery_bundle_has_required_scripts(tmp_path) -> None:
    config = Stage57RecoveryConfig(
        daemon_output_dir=str(tmp_path / "daemon"),
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "logs" / "daemon.pid"),
        log_file=str(tmp_path / "logs" / "daemon.log"),
        model_path="/models/qwen",
    )
    manifest = write_stage56_daemon_bundle(output_dir=config.daemon_output_dir, config=_make_daemon_config(config))
    validation = validate_stage56_daemon_bundle(output_dir=config.daemon_output_dir)
    assert validation["passes_stage_gate"]
    assert "start_script" in manifest["files"]
    assert "stop_script" in manifest["files"]
    assert "restart_script" in manifest["files"]


def test_stage57_detects_stale_pid_and_rotates_log(tmp_path) -> None:
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("99999999", encoding="utf-8")
    status = read_pid_status(pid_file)
    assert status["stale"]
    log_file = tmp_path / "daemon.log"
    log_file.write_text("x" * 64, encoding="utf-8")
    rotation = rotate_log_file(log_file, max_bytes=16, backup_count=2)
    assert rotation["rotated"]
    assert (tmp_path / "daemon.log.1").exists()


def test_stage57_port_preflight_detects_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        assert not port_available(host, port)
