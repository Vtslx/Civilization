# Cross-Device Encrypted Backup Replication Implementation Plan

> **For Codex:** Implement the manual replication path in the existing Civilization checkout; do not upload real state or configure a schedule without a writable target and explicit operational policy.

**Goal:** Copy a completed encrypted Orion backup to a caller-selected directory on a distinct filesystem and verify it before publishing.

**Architecture:** Add a pure persistence operation and a CLI wrapper. The operation authenticates the source bundle, checks that the target directory is existing/writable and on a different filesystem, streams ciphertext into a private temporary file on the target, compares SHA-256 and size, authenticates the temporary copy, and atomically publishes it without replacement. Passphrase handling follows the existing `CIVILIZATION_BACKUP_PASSPHRASE` convention.

**Tech Stack:** Python standard library, existing encrypted backup reader/writer, pytest.

---

## Task 1: Add API and CLI failure-path tests

**Files:** `tests/engine/test_memory_backup.py`, `src/civilization/cli.py`

1. Assert replication rejects a missing passphrase, a plaintext legacy bundle, a same-filesystem destination, and an existing destination file.
2. Assert the CLI requires `CIVILIZATION_BACKUP_PASSPHRASE` and exposes no `--passphrase` argument.
3. Assert failed verification leaves no temporary or final replica.

Use a dedicated `_filesystem_device_id` helper so unit tests can simulate a distinct target device without depending on host mount topology.

## Task 2: Implement streaming replication

**Files:** `src/civilization/engine/persistence/backup.py`, `src/civilization/engine/persistence/__init__.py`

Require the source to pass `verify_memory_backup` and report `encrypted=True`. Copy to a mode-0600 temporary file under the existing target directory, compare the source digest and file metadata before/after copying, verify the temporary bundle with the same passphrase, then atomically hard-link it to the requested final filename and fsync the destination directory. Clean up the temporary file on every failure. Return paths, bytes, digest, and verification state; never return secrets.

## Task 3: Add the CLI command

**Files:** `src/civilization/cli.py`, `tests/engine/test_memory_backup.py`

Add `civilization replicate-backup --backup FILE --destination-dir DIR [--json]`. Read the passphrase only from the existing environment variable, print a concise verified result, and leave retention/deletion out of this command.

Run the focused backup suite and the full configured repository tests; run compileall and `git diff --check`.

## Task 4: Perform a synthetic cross-device drill

The current writable internal data volume (`/tmp`) and iSCSI volume (`/Volumes/Net`) have different device IDs. Use synthetic test state on `/Volumes/Net`, a transient high-entropy passphrase, and a temporary destination under `/tmp`. Verify the encrypted replica and restore it to a new test directory. Remove both temporary trees after comparing all payload bytes. Do not use the production state directory.

**Result:** CLI copy, encrypted verification, and restore succeeded. Synthetic payload bytes matched exactly; source and target device IDs differed. The command returned only non-secret metadata. The passphrase and both temporary trees were removed at process exit.

The focused backup suite passed 28 tests; `compileall` and `git diff --check` passed. Full repository collection was unavailable because optional `torch`, `psutil`, and other development dependencies are missing. No dependencies were installed to bypass the environment gap.

Document that this validates cross-device integrity but is not offsite DR. `/Volumes/Recovery` is a non-writable system recovery volume, so it is not a backup target. Leave the feature without an automatic schedule until a writable offsite target, retention window, alerts, and restore policy are selected.
