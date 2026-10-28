# Jewelry Pair Acquisition Count Resolver

## Goal

Correct a jewelry-acquisition count when the answer audit counts the two earrings in one acquired pair as two separate jewelry items. Keep the behavior default-off and preserve the distinction between a pair acquired together and a question asking for individual earrings.

## Approaches considered

1. Rely on the existing answer/count-inventory prompt rule. Stage270 question `9d25d4e0` shows that the model can still emit quantity `2` for one acquired pair and produce a valid but incorrect total.
2. Ask a second model call to reinterpret the pair. This adds latency and another judgment that can disagree with exact user evidence.
3. Normalize the audited pair quantity deterministically. Chosen: adjust only a unique, valid, user-cited single-pair item when the question asks for pieces/items of jewelry and does not ask for individual earrings.

## Design

Add an opt-in resolver for jewelry-acquisition count questions. The original answer must pass the full count/citation validator. A corrected item must have quantity `2`, a piece-compatible unit, a description naming one pair of earrings, and a valid explicitly user-authored citation whose exact text describes acquiring one pair. Reject source text that names multiple pairs or lacks an acquisition action. Correct only that item's quantity to `1`, recompute the count, preserve all other items and citations, and validate the complete candidate before mutation. Use a concise count-only answer when the numeric total changes.

Questions about individual earrings, generic ownership, or an ambiguous/malformed inventory remain unchanged. Do not infer jewelry facts from assistant messages or from unstructured text without a user speaker marker.

## Acceptance and verification

- Correct the Stage270 `9d25d4e0` audit from four to three items while retaining its user citations.
- Leave individual-earring questions, weak/non-user source evidence, multiple-pair sources, and inconsistent audits unchanged.
- Keep the resolver off by default and record its status in answer/run metadata.
- Run the adapter suite, replay all Stage270 saved answers, live-judge each changed answer with seed 42, then run a 500-question candidate over frozen Stage268 retrieval and record paired outcomes and retrieval-order equality.
