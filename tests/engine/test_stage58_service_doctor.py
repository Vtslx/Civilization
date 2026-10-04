from __future__ import annotations

import json
from pathlib import Path

from civilization.engine.stages.stage58_service_doctor import (
    Stage58DoctorConfig,
    generate_stage58_runbook,
    run_stage58_doctor,
)


def test_stage58_generates_runbook_and_systemd_template(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    python_bin = repo / ".venv" / "bin" / "python"
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    output = tmp_path / "doctor"
    config = Stage58DoctorConfig(
        repo_root=str(repo),
        python_bin=str(python_bin),
        model_path=str(tmp_path / "model"),
        package_manifest="package.json",
        centroid_bundle="centroid.pt",
        daemon_output_dir=str(tmp_path / "daemon"),
        output_dir=str(output),
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "logs" / "daemon.pid"),
        log_file=str(tmp_path / "logs" / "daemon.log"),
    )
    manifest = generate_stage58_runbook(output_dir=output, config=config)
    unit = Path(manifest["files"]["systemd_unit"])
    assert unit.exists()
    assert "ExecStart=" in unit.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in unit.read_text(encoding="utf-8")
    assert Path(manifest["files"]["recover_script"]).exists()
    assert Path(manifest["files"]["doctor_script"]).exists()
    assert not manifest["installed_systemd"]


def test_stage58_doctor_reports_missing_model_remediation(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    py = repo / "python"
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    py.chmod(0o755)
    output = tmp_path / "doctor"
    config = Stage58DoctorConfig(
        repo_root=str(repo),
        python_bin=str(py),
        model_path=str(tmp_path / "missing-model"),
        package_manifest="missing-package.json",
        centroid_bundle="missing-centroid.pt",
        daemon_output_dir=str(tmp_path / "daemon"),
        output_dir=str(output),
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "logs" / "daemon.pid"),
        log_file=str(tmp_path / "logs" / "daemon.log"),
    )
    summary = run_stage58_doctor(output_dir=output, config=config)
    assert not summary["passes_stage_gate"]
    assert not summary["hard_checks"]["model_path"]
    assert any("model_path" in item for item in summary["remediation"])
    assert (output / "doctor_summary.json").exists()


def test_stage58_doctor_writes_machine_readable_json(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    py = repo / "python"
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    py.chmod(0o755)
    output = tmp_path / "doctor"
    config = Stage58DoctorConfig(
        repo_root=str(repo),
        python_bin=str(py),
        model_path=str(tmp_path / "model"),
        package_manifest="package.json",
        centroid_bundle="centroid.pt",
        daemon_output_dir=str(tmp_path / "daemon"),
        output_dir=str(output),
        log_dir=str(tmp_path / "logs"),
        pid_file=str(tmp_path / "logs" / "daemon.pid"),
        log_file=str(tmp_path / "logs" / "daemon.log"),
    )
    run_stage58_doctor(output_dir=output, config=config)
    payload = json.loads((output / "doctor_summary.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "stage58_service_doctor"
    assert "checks" in payload
    assert "hard_checks" in payload
    assert "remediation" in payload
