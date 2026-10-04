# Session memory durability

**Status:** implemented and verified
**Applies to:** every service started with `persist_sessions` enabled (the default)

## The problem

Session memory used to live only in the service process. Restarting the service
— a deploy, a crash, an OOM kill — lost every session. The global store had a
configured file path, but nothing was written to it, so a restart also lost any
cross-session state.

## The design

Reads stay in memory; writes become durable. Concretely:

```
request ──► in-memory store ──► (read path, unchanged: retrieval, ranking, injection)
                  │
                  └─ mutation ──► journal append (os.write, append-only, per session)
                                        │
                            ┌───────────┴────────────┐
                            │ commit by fsync policy │  interval | every_write | never
                            └───────────┬────────────┘
                                        │
                     compaction (every N records) ──► snapshot.json + journal reset
                                        │
                    startup / first touch ──► hydrate = snapshot + journal tail
```

Four properties make recovery obviously correct:

1. **The append leaves the process immediately.** Records are written with
   `os.write` on an `O_APPEND` descriptor. An abrupt *process* death therefore
   loses nothing that was already acknowledged — the bytes are in the operating
   system's page cache.
2. **The commit policy bounds the loss window.** `fsync` is called by a
   background thread every `fsync_interval_seconds` (default 0.25 s), on every
   write, or never. A *power* loss loses at most the uncommitted tail.
3. **Every record is an idempotent upsert.** Replaying a journal on top of a
   snapshot cannot duplicate or corrupt state, so a crash between "snapshot
   written" and "journal truncated" is harmless.
4. **A torn final line is ignored.** A partial line from a crash mid-write is
   detected and dropped; every complete record before it is kept.

### What is durable, and when

| Event | Effect | Why |
|---|---|---|
| Clean shutdown | nothing lost | `close()` commits and flushes |
| Service restarts (deploy, upgrade) | nothing lost | journals are on disk, replayed on startup |
| Process killed (`kill -9`, OOM) | nothing acknowledged is lost | records already left the process via `os.write` |
| Machine power loss / hard reset | loses at most the uncommitted tail | bounded by the fsync policy (default ≤ 0.25 s of writes) |
| Disk failure | local state is lost | there is no replication; use a volume with its own durability |

### The optional persistence sidecar

`civilization persist-sidecar` runs the journal and commit loop in a **separate
process**; the service sends records over a UNIX socket.

What it buys: fsync work happens outside the service process, and the durable
stream stays open across service restarts, so a restart cannot interrupt a
commit cycle.

What it does **not** buy: extra data safety. Durability comes from the journal
plus the fsync policy — not from which process performs the write. Both
processes are on the same machine, so a power loss behaves the same way.

If the sidecar is unreachable (not started, crashed, socket path too long), the
service **does not fail memory writes**: it degrades to local in-process
journaling in the same state directory and reports `degraded: true` with the
reason in the status payload. Re-attaching requires a service restart.

## Configuration

`EmbeddedConfig` fields (and the matching `civilization serve` flags):

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `persist_sessions` | `--persist-sessions / --no-persist-sessions` | `True` | journal session memory under `state_dir` |
| `persistence_mode` | `--persistence-mode {in_process,sidecar}` | `in_process` | who owns the files |
| `fsync_policy` | `--fsync {every_write,interval,never}` | `interval` | commit policy |
| `fsync_interval_seconds` | `--fsync-interval` | `0.25` | commit interval |
| `snapshot_every_records` | `--snapshot-every` | `512` | journal compaction threshold |
| `sidecar_socket` | `--sidecar-socket` | none | sidecar endpoint, required in sidecar mode |
| `persist_traces` | `--persist-traces` | `False` | also journal memory trace events |

Layout under the state directory:

```
<state_dir>/
  sessions/
    <session_id>/
      journal.ndjson     append-only records
      snapshot.json      compacted state
  var/                 jobs, results, exports (unchanged)
```

## Operating

```bash
# inspect: sessions on disk, pending records, commit watermark, mode, degraded flag
curl -s localhost:8765/admin/orion/memory | jq '.persistence'

# force a commit (for example before a snapshot of the volume)
curl -s -X POST localhost:8765/admin/orion/memory/flush | jq '.persistence.sink'

# run with a sidecar
civilization persist-sidecar --state-dir var/state --socket /tmp/civilization.sock &
civilization serve --persistence-mode sidecar --sidecar-socket /tmp/civilization.sock \
  --provider-base-url <url> --provider-model <model>

# strongest durability, highest write cost
civilization serve --fsync every_write --provider-base-url <url> --provider-model <model>
```

Monitor `persistence.sink.pending_records`: it is the number of acknowledged
records not yet committed. A number that keeps growing means the disk cannot
keep up with the write rate.

## Verified behaviour

Automated coverage (`tests/engine/test_session_persistence.py`):

- a session written before a restart is readable after it, including links and
  cell updates, and id counters continue instead of colliding;
- a torn final journal line is ignored while the preceding records survive;
- replaying stale journal records on top of a snapshot neither duplicates cells
  nor links;
- `every_write` and `interval` commit policies both reach `pending_records == 0`;
- the sidecar receives records, drains them when the service dies abruptly, and
  the state is recovered from the files afterwards;
- an unreachable sidecar degrades to local writing instead of failing writes;
- hostile session ids cannot escape the state directory.

End-to-end crash test, run against a service with `fsync=interval`
(0.1 s interval):

1. service writes 3 memory cells into one session;
2. the process is killed with `kill -9` — no clean shutdown, no flush call;
3. a new service process starts on the same state directory;
4. `read_memory` returns all 3 cells, with the same `cell_id`s.

## Limits and non-goals

- **No replication.** Durability here means "survives a restart or a crash on
  this machine", not "survives losing the disk". Point `state_dir` at a volume
  that has its own redundancy if you need that.
- **Global store keeps its explicit lifecycle.** `POST
  /admin/orion/global-store/save|load` still writes and reads its own file
  deliberately; the journal covers session memory.
- **Traces are off by default.** Cells and links are the state that matters for
  decisions; the diagnostic trace stream is opt-in (`persist_traces`).
- **One writer per state directory.** Two services sharing a directory would
  interleave journals. Use the sidecar mode or separate directories.
- **No compression or retention policy yet.** Journals grow with writes; the
  snapshot bounds the *replay* cost, not the byte count. A retention pass over
  old sessions is future work.
