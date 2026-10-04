# Reproduction scripts

## `memory_ab.py`

Paired A/B of Civilization native (Orion) memory: `read_memory = true` vs.
`read_memory = false`, everything else held constant.

The script is self-contained. It embeds the 25 synthetic question fixtures and
needs only:

- a reachable Civilization service (`GET /ready` must report ready), and
- the `civilization` SDK importable (from this repository or installed).

```bash
# from the repository root
python -m pip install -e '.[embedded]'     # or: pip install -e .

cd docs/versions/v0.00.07-helix/experiments/scripts

# full protocol: 3 rounds × 25 questions → 150 predictions
CIVILIZATION_LOCAL_BASE_URL=http://127.0.0.1:8765 \
  python memory_ab.py --rounds 3 --provider-model <model-name> --output memory_on_vs_off.json

# smoke run: 1 round, first 4 questions
python memory_ab.py --rounds 1 --max-cases 4 --output smoke.json
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--base-url` | `$CIVILIZATION_LOCAL_BASE_URL` or `http://127.0.0.1:8765` | service endpoint |
| `--provider-model` | `$CIVILIZATION_PROVIDER_MODEL` or `not-recorded` | recorded in the result for provenance |
| `--rounds` | `3` | repetitions of the fixed question list |
| `--max-cases` | `0` (all) | limit the question list for smoke runs |
| `--timeout` | `180` | per-request timeout in seconds |
| `--output` | `./memory_on_vs_off_<run_id>.json` | result path |

Behaviour worth knowing before you run it:

- Each run writes memory into **fresh sessions named after its run id**, so runs
  never share memory with each other or with your application sessions.
- `write_memory` is false in both conditions, so the answer key cannot drift
  during a round.
- Per-case predictions, injected cell ids, retrieval traces, latencies, and
  errors are all recorded in the result JSON; a failed call is recorded, never
  masked.
- The question fixtures are synthetic and carry no personal or customer data.

Reading the output: compare the `paired` block (wins/losses) rather than only
the accuracy delta, and read the `limitations` list in the same file before
quoting any number.

## Published recorded run

[../data/memory-on-off-ab-recorded.json](../data/memory-on-off-ab-recorded.json)
is the recorded 3-round run referenced throughout
[../memory-on-off-ab.md](../memory-on-off-ab.md): 150 case rows, run id
`20261003T213755Z-ab-fae54b`, implementation revision
`9ebb73034c10f123e31ee9222ea82464d32967b5`.
