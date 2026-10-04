#!/usr/bin/env python3
"""Paired A/B of Civilization native (Orion) memory: memory on vs. memory off.

One controlled comparison: the same question, the same four options, the same
task name, the same session memory contents, the same provider model, and the
same seed. The only field that differs is ``CivilizationRequest.read_memory``.
``write_memory`` is false in both conditions so the answer key stays fixed.

The questions are synthetic four-choice items built for this experiment; the
fixtures are embedded below, so a run needs nothing but a reachable service and
the SDK.

Usage:

    CIVILIZATION_LOCAL_BASE_URL=http://127.0.0.1:8765 \\
      python memory_ab.py --rounds 3 --output memory_on_vs_off.json

    # quick smoke instead of the full protocol
    python memory_ab.py --rounds 1 --max-cases 4 --output smoke.json

The service must be ready (``GET /ready``) with a decision backend configured.
Each run writes its memory into fresh sessions named after the run id, so
repeated runs never share memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from civilization import CivilizationClient, CivilizationRequest

SEED = 20261004
DEFAULT_ROUNDS = 3

CASES: list[dict[str, object]] = [{'case_id': 'fact-orbit-17',
  'axis': 'atomic_fact_recall',
  'question': 'What archive seal is recorded for evaluation case ORBIT-17?',
  'correct': 'plum-kestrel-614',
  'distractors': ['amber-willow-208', 'velvet-puma-731', 'coral-ibis-904'],
  'records': [{'role': 'target',
               'content': 'For evaluation case ORBIT-17, the archive seal is plum-kestrel-614.',
               'summary': 'ORBIT-17 archive seal: plum-kestrel-614',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_atomic_fact_recall'},
 {'case_id': 'fact-fable-29',
  'axis': 'atomic_fact_recall',
  'question': 'What handoff token is recorded for evaluation case FABLE-29?',
  'correct': 'moss-cipher-271',
  'distractors': ['silver-heron-482', 'copper-lynx-530', 'indigo-tern-116'],
  'records': [{'role': 'target',
               'content': 'For evaluation case FABLE-29, the handoff token is moss-cipher-271.',
               'summary': 'FABLE-29 handoff token: moss-cipher-271',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_atomic_fact_recall'},
 {'case_id': 'fact-cirrus-08',
  'axis': 'atomic_fact_recall',
  'question': 'What route code is recorded for evaluation case CIRRUS-08?',
  'correct': 'saffron-otter-930',
  'distractors': ['violet-fox-219', 'birch-eagle-644', 'jade-marten-308'],
  'records': [{'role': 'target',
               'content': 'For evaluation case CIRRUS-08, the route code is saffron-otter-930.',
               'summary': 'CIRRUS-08 route code: saffron-otter-930',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_atomic_fact_recall'},
 {'case_id': 'fact-lattice-51',
  'axis': 'atomic_fact_recall',
  'question': 'What batch marker is recorded for evaluation case LATTICE-51?',
  'correct': 'quartz-finch-128',
  'distractors': ['cedar-lantern-603', 'onyx-raven-857', 'pearl-wren-402'],
  'records': [{'role': 'target',
               'content': 'For evaluation case LATTICE-51, the batch marker is quartz-finch-128.',
               'summary': 'LATTICE-51 batch marker: quartz-finch-128',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_atomic_fact_recall'},
 {'case_id': 'fact-vesper-42',
  'axis': 'atomic_fact_recall',
  'question': 'What checkpoint key is recorded for evaluation case VESPER-42?',
  'correct': 'cobalt-anchovy-743',
  'distractors': ['maple-badger-365', 'ruby-swallow-924', 'flint-hare-051'],
  'records': [{'role': 'target',
               'content': 'For evaluation case VESPER-42, the checkpoint key is cobalt-anchovy-743.',
               'summary': 'VESPER-42 checkpoint key: cobalt-anchovy-743',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_atomic_fact_recall'},
 {'case_id': 'preference-profile-11',
  'axis': 'preference_recall',
  'question': 'According to preference profile PROFILE-11, what is the preferred report format?',
  'correct': 'two-column Markdown table',
  'distractors': ['numbered checklist', 'plain-text paragraph', 'JSON object'],
  'records': [{'role': 'target',
               'content': 'For synthetic preference profile PROFILE-11, the preferred report format is two-column '
                          'Markdown table.',
               'summary': 'PROFILE-11 preferred report format: two-column Markdown table',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_preference_recall'},
 {'case_id': 'preference-profile-22',
  'axis': 'preference_recall',
  'question': 'According to preference profile PROFILE-22, what is the preferred summary length?',
  'correct': 'exactly three bullet points',
  'distractors': ['one sentence', 'five bullet points', 'a full-page narrative'],
  'records': [{'role': 'target',
               'content': 'For synthetic preference profile PROFILE-22, the preferred summary length is exactly '
                          'three bullet points.',
               'summary': 'PROFILE-22 preferred summary length: exactly three bullet points',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_preference_recall'},
 {'case_id': 'preference-profile-33',
  'axis': 'preference_recall',
  'question': 'According to preference profile PROFILE-33, what is the preferred reminder time?',
  'correct': '09:15 local time',
  'distractors': ['07:30 local time', '12:45 local time', '18:00 local time'],
  'records': [{'role': 'target',
               'content': 'For synthetic preference profile PROFILE-33, the preferred reminder time is 09:15 local '
                          'time.',
               'summary': 'PROFILE-33 preferred reminder time: 09:15 local time',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_preference_recall'},
 {'case_id': 'preference-profile-44',
  'axis': 'preference_recall',
  'question': 'According to preference profile PROFILE-44, what is the preferred chart style?',
  'correct': 'small-multiples line charts',
  'distractors': ['single stacked bar chart', 'one large pie chart', 'plain data table'],
  'records': [{'role': 'target',
               'content': 'For synthetic preference profile PROFILE-44, the preferred chart style is small-multiples '
                          'line charts.',
               'summary': 'PROFILE-44 preferred chart style: small-multiples line charts',
               'memory_system': 'semantic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_preference_recall'},
 {'case_id': 'temporal-timeline-01',
  'axis': 'temporal_ordering',
  'question': 'For timeline TIMELINE-01, which event occurred earliest? Choose from the named events.',
  'correct': 'review',
  'distractors': ['intake', 'rollout', 'archive'],
  'records': [{'role': 'target',
               'content': 'For synthetic timeline TIMELINE-01: intake occurred 2024-05-12; review occurred '
                          '2024-05-09; rollout occurred 2024-06-01.',
               'summary': 'TIMELINE-01 event dates: intake occurred 2024-05-12; review occurred 2024-05-09; rollout '
                          'occurred 2024-06-01',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_temporal_ordering'},
 {'case_id': 'temporal-timeline-02',
  'axis': 'temporal_ordering',
  'question': 'For timeline TIMELINE-02, which event occurred latest? Choose from the named events.',
  'correct': 'setup',
  'distractors': ['validation', 'freeze', 'archive'],
  'records': [{'role': 'target',
               'content': 'For synthetic timeline TIMELINE-02: setup occurred 2025-01-22; validation occurred '
                          '2025-01-09; freeze occurred 2025-01-13.',
               'summary': 'TIMELINE-02 event dates: setup occurred 2025-01-22; validation occurred 2025-01-09; '
                          'freeze occurred 2025-01-13',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_temporal_ordering'},
 {'case_id': 'temporal-timeline-03',
  'axis': 'temporal_ordering',
  'question': 'For timeline TIMELINE-03, which event occurred earliest? Choose from the named events.',
  'correct': 'handoff',
  'distractors': ['design', 'audit', 'archive'],
  'records': [{'role': 'target',
               'content': 'For synthetic timeline TIMELINE-03: design occurred 2023-07-02; audit occurred '
                          '2023-07-04; handoff occurred 2023-06-30.',
               'summary': 'TIMELINE-03 event dates: design occurred 2023-07-02; audit occurred 2023-07-04; handoff '
                          'occurred 2023-06-30',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_temporal_ordering'},
 {'case_id': 'temporal-timeline-04',
  'axis': 'temporal_ordering',
  'question': 'For timeline TIMELINE-04, which event occurred latest? Choose from the named events.',
  'correct': 'launch',
  'distractors': ['staging', 'approval', 'archive'],
  'records': [{'role': 'target',
               'content': 'For synthetic timeline TIMELINE-04: staging occurred 2026-02-10; approval occurred '
                          '2026-02-01; launch occurred 2026-02-15.',
               'summary': 'TIMELINE-04 event dates: staging occurred 2026-02-10; approval occurred 2026-02-01; '
                          'launch occurred 2026-02-15',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_temporal_ordering'},
 {'case_id': 'interference-project-ax14',
  'axis': 'similar_entity_interference',
  'question': 'What retention code belongs to PROJECT-AX14 (not the similarly named PROJECT-AX41)?',
  'correct': 'violet-fox-219',
  'distractors': ['amber-fox-219', 'birch-eagle-644', 'jade-marten-308'],
  'records': [{'role': 'target',
               'content': 'For entity PROJECT-AX14, the retention code is violet-fox-219.',
               'summary': 'PROJECT-AX14 retention code: violet-fox-219',
               'memory_system': 'episodic'},
              {'role': 'neighbor',
               'content': 'For nearby entity PROJECT-AX41, the retention code is amber-fox-219.',
               'summary': 'PROJECT-AX41 retention code: amber-fox-219',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_similar_entity_interference'},
 {'case_id': 'interference-device-l09',
  'axis': 'similar_entity_interference',
  'question': 'What reset phrase belongs to DEVICE-L09 (not the similarly named DEVICE-L90)?',
  'correct': 'cedar-lantern-603',
  'distractors': ['cedar-lantern-630', 'onyx-raven-857', 'pearl-wren-402'],
  'records': [{'role': 'target',
               'content': 'For entity DEVICE-L09, the reset phrase is cedar-lantern-603.',
               'summary': 'DEVICE-L09 reset phrase: cedar-lantern-603',
               'memory_system': 'episodic'},
              {'role': 'neighbor',
               'content': 'For nearby entity DEVICE-L90, the reset phrase is cedar-lantern-630.',
               'summary': 'DEVICE-L90 reset phrase: cedar-lantern-630',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_similar_entity_interference'},
 {'case_id': 'interference-queue-m27',
  'axis': 'similar_entity_interference',
  'question': 'What routing label belongs to QUEUE-M27 (not the similarly named QUEUE-M72)?',
  'correct': 'maple-badger-365',
  'distractors': ['maple-badger-536', 'ruby-swallow-924', 'flint-hare-051'],
  'records': [{'role': 'target',
               'content': 'For entity QUEUE-M27, the routing label is maple-badger-365.',
               'summary': 'QUEUE-M27 routing label: maple-badger-365',
               'memory_system': 'episodic'},
              {'role': 'neighbor',
               'content': 'For nearby entity QUEUE-M72, the routing label is maple-badger-536.',
               'summary': 'QUEUE-M72 routing label: maple-badger-536',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_similar_entity_interference'},
 {'case_id': 'interference-vault-r36',
  'axis': 'similar_entity_interference',
  'question': 'What recovery tag belongs to VAULT-R36 (not the similarly named VAULT-R63)?',
  'correct': 'silver-heron-482',
  'distractors': ['silver-heron-824', 'copper-lynx-530', 'indigo-tern-116'],
  'records': [{'role': 'target',
               'content': 'For entity VAULT-R36, the recovery tag is silver-heron-482.',
               'summary': 'VAULT-R36 recovery tag: silver-heron-482',
               'memory_system': 'episodic'},
              {'role': 'neighbor',
               'content': 'For nearby entity VAULT-R63, the recovery tag is silver-heron-824.',
               'summary': 'VAULT-R63 recovery tag: silver-heron-824',
               'memory_system': 'episodic'}],
  'target_roles': ['target'],
  'task_name': 'ordinary_memory_similar_entity_interference'},
 {'case_id': 'update-update-01',
  'axis': 'knowledge_update',
  'question': 'After the 2026-09-20 update, what is the current zone code for UPDATE-01?',
  'correct': 'teal-bison',
  'distractors': ['amber-fox', 'retired-update-01', 'pending-update-01'],
  'records': [{'role': 'old',
               'content': 'Before the 2026-09-20 update, the zone code for UPDATE-01 was amber-fox; this value is '
                          'superseded.',
               'summary': 'Historical UPDATE-01 zone code: amber-fox; superseded 2026-09-20',
               'memory_system': 'episodic'},
              {'role': 'current',
               'content': 'The 2026-09-20 knowledge update replaced amber-fox. The current zone code for UPDATE-01 '
                          'is teal-bison, effective 2026-09-20.',
               'summary': 'Current UPDATE-01 zone code: teal-bison; effective 2026-09-20',
               'memory_system': 'episodic'}],
  'target_roles': ['current'],
  'task_name': 'ordinary_memory_knowledge_update'},
 {'case_id': 'update-update-02',
  'axis': 'knowledge_update',
  'question': 'After the 2026-09-20 update, what is the current routing label for UPDATE-02?',
  'correct': 'route-jade',
  'distractors': ['route-copper', 'retired-update-02', 'pending-update-02'],
  'records': [{'role': 'old',
               'content': 'Before the 2026-09-20 update, the routing label for UPDATE-02 was route-copper; this '
                          'value is superseded.',
               'summary': 'Historical UPDATE-02 routing label: route-copper; superseded 2026-09-20',
               'memory_system': 'episodic'},
              {'role': 'current',
               'content': 'The 2026-09-20 knowledge update replaced route-copper. The current routing label for '
                          'UPDATE-02 is route-jade, effective 2026-09-20.',
               'summary': 'Current UPDATE-02 routing label: route-jade; effective 2026-09-20',
               'memory_system': 'episodic'}],
  'target_roles': ['current'],
  'task_name': 'ordinary_memory_knowledge_update'},
 {'case_id': 'update-update-03',
  'axis': 'knowledge_update',
  'question': 'After the 2026-09-20 update, what is the current owner group for UPDATE-03?',
  'correct': 'group-south',
  'distractors': ['group-north', 'retired-update-03', 'pending-update-03'],
  'records': [{'role': 'old',
               'content': 'Before the 2026-09-20 update, the owner group for UPDATE-03 was group-north; this value '
                          'is superseded.',
               'summary': 'Historical UPDATE-03 owner group: group-north; superseded 2026-09-20',
               'memory_system': 'episodic'},
              {'role': 'current',
               'content': 'The 2026-09-20 knowledge update replaced group-north. The current owner group for '
                          'UPDATE-03 is group-south, effective 2026-09-20.',
               'summary': 'Current UPDATE-03 owner group: group-south; effective 2026-09-20',
               'memory_system': 'episodic'}],
  'target_roles': ['current'],
  'task_name': 'ordinary_memory_knowledge_update'},
 {'case_id': 'update-update-04',
  'axis': 'knowledge_update',
  'question': 'After the 2026-09-20 update, what is the current retention tier for UPDATE-04?',
  'correct': 'tier-two',
  'distractors': ['tier-seven', 'retired-update-04', 'pending-update-04'],
  'records': [{'role': 'old',
               'content': 'Before the 2026-09-20 update, the retention tier for UPDATE-04 was tier-seven; this value '
                          'is superseded.',
               'summary': 'Historical UPDATE-04 retention tier: tier-seven; superseded 2026-09-20',
               'memory_system': 'episodic'},
              {'role': 'current',
               'content': 'The 2026-09-20 knowledge update replaced tier-seven. The current retention tier for '
                          'UPDATE-04 is tier-two, effective 2026-09-20.',
               'summary': 'Current UPDATE-04 retention tier: tier-two; effective 2026-09-20',
               'memory_system': 'episodic'}],
  'target_roles': ['current'],
  'task_name': 'ordinary_memory_knowledge_update'},
 {'case_id': 'isolation-secure-01',
  'axis': 'session_isolation_and_abstention',
  'question': 'What deployment token belongs to SECURE-01? This request is in another session. If no matching memory '
              'is visible here, choose NOT IN SESSION instead of guessing.',
  'correct': 'NOT IN SESSION',
  'distractors': ['hidden-puma-738', 'quiet-ibis-205', 'silver-mole-461'],
  'records': [{'role': 'source_only',
               'content': 'For isolated record SECURE-01, the deployment token is hidden-puma-738.',
               'summary': 'SECURE-01 deployment token: hidden-puma-738',
               'memory_system': 'episodic'}],
  'target_roles': [],
  'task_name': 'ordinary_memory_session_isolation_and_abstention'},
 {'case_id': 'isolation-secure-02',
  'axis': 'session_isolation_and_abstention',
  'question': 'What ledger tag belongs to SECURE-02? This request is in another session. If no matching memory is '
              'visible here, choose NOT IN SESSION instead of guessing.',
  'correct': 'NOT IN SESSION',
  'distractors': ['cinder-wren-592', 'amber-hare-183', 'blue-heron-740'],
  'records': [{'role': 'source_only',
               'content': 'For isolated record SECURE-02, the ledger tag is cinder-wren-592.',
               'summary': 'SECURE-02 ledger tag: cinder-wren-592',
               'memory_system': 'episodic'}],
  'target_roles': [],
  'task_name': 'ordinary_memory_session_isolation_and_abstention'},
 {'case_id': 'isolation-secure-03',
  'axis': 'session_isolation_and_abstention',
  'question': 'What recovery phrase belongs to SECURE-03? This request is in another session. If no matching memory '
              'is visible here, choose NOT IN SESSION instead of guessing.',
  'correct': 'NOT IN SESSION',
  'distractors': ['frost-otter-316', 'jade-fox-804', 'copper-tern-259'],
  'records': [{'role': 'source_only',
               'content': 'For isolated record SECURE-03, the recovery phrase is frost-otter-316.',
               'summary': 'SECURE-03 recovery phrase: frost-otter-316',
               'memory_system': 'episodic'}],
  'target_roles': [],
  'task_name': 'ordinary_memory_session_isolation_and_abstention'},
 {'case_id': 'isolation-secure-04',
  'axis': 'session_isolation_and_abstention',
  'question': 'What audit nonce belongs to SECURE-04? This request is in another session. If no matching memory is '
              'visible here, choose NOT IN SESSION instead of guessing.',
  'correct': 'NOT IN SESSION',
  'distractors': ['moss-raven-627', 'pearl-lynx-913', 'violet-badger-452'],
  'records': [{'role': 'source_only',
               'content': 'For isolated record SECURE-04, the audit nonce is moss-raven-627.',
               'summary': 'SECURE-04 audit nonce: moss-raven-627',
               'memory_system': 'episodic'}],
  'target_roles': [],
  'task_name': 'ordinary_memory_session_isolation_and_abstention'}]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q + 0.5)))
    return ordered[index]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate both conditions and the paired outcomes."""

    result: dict[str, object] = {}
    for condition in ("memory_on", "memory_off"):
        group = [row for row in rows if row["condition"] == condition]
        answerable = [row for row in group if row["axis"] != "session_isolation_and_abstention"]
        isolation = [row for row in group if row["axis"] == "session_isolation_and_abstention"]
        latencies = [float(row["latency_ms"]) for row in group if row["error"] is None]
        correct = sum(bool(row["exact_match"]) for row in group)
        result[condition] = {
            "n": len(group),
            "exact_matches": correct,
            "accuracy": correct / len(group) if group else None,
            "answerable_n": len(answerable),
            "answerable_correct": sum(bool(row["exact_match"]) for row in answerable),
            "answerable_accuracy": (
                sum(bool(row["exact_match"]) for row in answerable) / len(answerable) if answerable else None
            ),
            "isolation_n": len(isolation),
            "isolation_abstentions_correct": sum(bool(row["exact_match"]) for row in isolation),
            "api_errors": sum(row["error"] is not None for row in group),
            "target_cell_in_context": sum(bool(row["target_injected"]) for row in answerable),
            "target_injection_rate": (
                sum(bool(row["target_injected"]) for row in answerable) / len(answerable) if answerable else None
            ),
            "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
            "p50_latency_ms": round(percentile(latencies, 0.50), 3) if latencies else None,
            "p95_latency_ms": round(percentile(latencies, 0.95), 3) if latencies else None,
        }

    paired: dict[tuple[int, str], dict[str, bool]] = {}
    for row in rows:
        paired.setdefault((int(row["round"]), str(row["case_id"])), {})[str(row["condition"])] = bool(row["exact_match"])
    outcomes = {"both_correct": 0, "memory_only_correct": 0, "memory_off_only_correct": 0, "both_wrong": 0}
    by_axis: dict[str, dict[str, int]] = {}
    axis_for_case = {str(row["case_id"]): str(row["axis"]) for row in rows}
    for (_, case_id), values in paired.items():
        if set(values) != {"memory_on", "memory_off"}:
            continue
        on, off = values["memory_on"], values["memory_off"]
        if on and off:
            bucket = "both_correct"
        elif on:
            bucket = "memory_only_correct"
        elif off:
            bucket = "memory_off_only_correct"
        else:
            bucket = "both_wrong"
        outcomes[bucket] += 1
        by_axis.setdefault(axis_for_case[case_id], {key: 0 for key in outcomes})[bucket] += 1

    on_summary = result["memory_on"]
    off_summary = result["memory_off"]
    result["paired"] = {
        **outcomes,
        "pairs": sum(outcomes.values()),
        "net_memory_wins": outcomes["memory_only_correct"] - outcomes["memory_off_only_correct"],
        "accuracy_delta_percentage_points": round(
            (float(on_summary["accuracy"]) - float(off_summary["accuracy"])) * 100, 2
        ),
        "answerable_accuracy_delta_percentage_points": round(
            (float(on_summary["answerable_accuracy"]) - float(off_summary["answerable_accuracy"])) * 100, 2
        ),
        "by_axis": by_axis,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("CIVILIZATION_LOCAL_BASE_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--provider-model", default=os.environ.get("CIVILIZATION_PROVIDER_MODEL", "not-recorded"))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-cases", type=int, default=0, help="limit the case list for smoke runs; 0 = all cases")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default=None, help="result JSON path (default: ./memory_on_vs_off_<run_id>.json)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-ab-" + uuid4().hex[:6]
    cases = CASES[: args.max_cases] if args.max_cases else CASES

    client = CivilizationClient(args.base_url, timeout=args.timeout)
    ready = client.ready()
    ready_value = ready.get("ready", ready.get("status"))
    if ready_value not in (True, "ready", "ok"):
        raise RuntimeError(f"Civilization service is not ready: {ready_value!r}")

    prepared: list[dict[str, object]] = []
    rng = random.Random(SEED)
    for case in cases:
        options = [str(case["correct"]), *[str(value) for value in case["distractors"]]]
        rng.shuffle(options)
        prompt = str(case["question"])
        is_isolation = case["axis"] == "session_isolation_and_abstention"
        if not is_isolation:
            prompt += " Answer from the memory available in this session."
        prepared.append({"case": case, "options": options, "prompt": prompt, "isolation": is_isolation})

    rows: list[dict[str, object]] = []
    for round_number in range(1, args.rounds + 1):
        session_suffix = f"{run_id}-{round_number}"
        memory_session = f"ab-memory-{session_suffix}"
        source_session = f"ab-source-{session_suffix}"
        target_session = f"ab-target-{session_suffix}"
        seeded: dict[str, dict[str, str]] = {}
        for item in prepared:
            case = item["case"]
            case_id = str(case["case_id"])
            write_session = source_session if item["isolation"] else memory_session
            seeded[case_id] = {}
            for record in case["records"]:
                cell = client.write_memory(
                    session_id=write_session,
                    memory_system=record["memory_system"],
                    content=record["content"],
                    summary=record["summary"],
                    confidence=1.0,
                    importance=1.0,
                    metadata={
                        "evaluation_run_id": run_id,
                        "round": round_number,
                        "case_id": case_id,
                        "role": record["role"],
                    },
                )
                if not cell.get("cell_id"):
                    raise RuntimeError(f"Memory write returned no cell id for {case_id}/{record['role']}")
                seeded[case_id][record["role"]] = str(cell["cell_id"])

        for index, item in enumerate(prepared):
            case = item["case"]
            case_id = str(case["case_id"])
            axis = str(case["axis"])
            is_isolation = bool(item["isolation"])
            session_id = target_session if is_isolation else memory_session
            target_ids = [seeded[case_id][role] for role in case["target_roles"]]
            order = ("memory_on", "memory_off") if (index + round_number) % 2 else ("memory_off", "memory_on")
            results: dict[str, dict[str, object]] = {}
            for condition in order:
                request = CivilizationRequest(
                    request_id=f"{run_id}-r{round_number}-{case_id}-{condition}",
                    text=str(item["prompt"]),
                    answer_options=list(item["options"]),
                    session_id=session_id,
                    task_name=str(case["task_name"]),
                    seed=SEED,
                    read_memory=condition == "memory_on",
                    write_memory=False,
                )
                started = time.perf_counter()
                predicted: str | None = None
                trace: dict[str, object] = {}
                error: str | None = None
                try:
                    prediction = client.predict(request)
                    predicted = prediction.option_text
                    trace = dict(prediction.memory_trace)
                except Exception as exc:  # noqa: BLE001 - recorded per case, never masked
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started) * 1000
                injected_raw = trace.get("retrieved_cell_ids", [])
                injected_ids = [str(value) for value in injected_raw] if isinstance(injected_raw, list) else []
                results[condition] = {
                    "predicted": predicted,
                    "exact_match": predicted == case["correct"] if predicted is not None else False,
                    "injected_ids": injected_ids,
                    "target_injected": any(cell_id in target_ids for cell_id in injected_ids),
                    "injected_item_count": int(trace.get("injected_item_count", len(injected_ids))),
                    "latency_ms": round(latency_ms, 3),
                    "error": error,
                }
            for condition in ("memory_on", "memory_off"):
                item_result = results[condition]
                rows.append(
                    {
                        "round": round_number,
                        "case_id": case_id,
                        "axis": axis,
                        "condition": condition,
                        "session_scope": "isolated_negative_control" if is_isolation else "same_session_recall",
                        "question": case["question"],
                        "options": item["options"],
                        "expected": case["correct"],
                        "predicted": item_result["predicted"],
                        "exact_match": item_result["exact_match"],
                        "target_cell_ids": target_ids,
                        "injected_cell_ids": item_result["injected_ids"],
                        "target_injected": item_result["target_injected"],
                        "injected_item_count": item_result["injected_item_count"],
                        "latency_ms": item_result["latency_ms"],
                        "error": item_result["error"],
                    }
                )

    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_model": args.provider_model,
        "base_url": args.base_url,
        "rounds": args.rounds,
        "unique_questions_per_round": len(prepared),
        "paired_design": (
            "Same text, answer options, task name, session memory contents, provider model, and seed; "
            "only CivilizationRequest.read_memory differs. write_memory=False in both conditions. "
            "Condition order alternates by case."
        ),
        "scoring": "Exact match to a synthetic answer key; no LLM judge.",
        "summary": summarize(rows),
        "limitations": [
            "The three repeats measure repeatability of the same questions, not independent question samples.",
            "The ordinary prediction endpoint receives fixed-choice questions, not free-form conversation or "
            "automatic memory extraction.",
            "Session store persistence across process restart and global-memory behavior are outside this comparison.",
            "No-memory calls can still guess from the question or option wording; read paired wins and losses, "
            "not only the accuracy delta.",
        ],
        "cases": rows,
    }
    output_path = Path(args.output) if args.output else Path.cwd() / f"memory_on_vs_off_{run_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result_path": str(output_path), "run_id": run_id, "summary": result["summary"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
