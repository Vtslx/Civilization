# Jewelry Pair Acquisition Count Resolver

## Goal

Correct a jewelry-acquisition count when the structured audit counts both earrings in one acquired pair as two separate jewelry items. The resolver remains default-off and does not change questions that ask for individual earrings.

## Approaches considered

1. Rely on the existing generation/count-inventory prompt rule. Stage270 row `9d25d4e0` shows that the model can still assign quantity `2` to one acquired pair and return a valid but incorrect total.
2. Ask another model call to reinterpret the pair. This adds latency and another stochastic judgment.
3. Normalize a uniquely cited pair item deterministically. Chosen: it follows the existing domain rule, uses exact user evidence, and can fail closed when the source or item audit is ambiguous.

## Design

Add an opt-in resolver only for numeric acquisition questions phrased as pieces/items of jewelry. The original audit must pass full citation and count validation. A correction requires a single-pair-of-earrings item with quantity `2`, a piece-compatible unit, and an exact user-authored citation in which one pair is acquired together. Reject citations describing multiple pairs, assistant-only statements, missing acquisition verbs, cumulative/aggregate rows, and requests for individual earrings. Adjust the pair quantity to `1`, recompute the total, preserve all other items and citations, and validate the candidate before mutating the answer.

When the count changes, produce a concise count-only answer. The resolver does not infer new jewelry, change dates, or modify persistent memory.

## Acceptance and verification

- Correct Stage270 row `9d25d4e0` from four to three pieces using the user-cited emerald-earring pair.
- Leave individual-earring questions, weak/missing acquisition evidence, multiple-pair sources, and invalid audits unchanged.
- Keep the feature off by default and record its run and per-question audit status.
- Run the adapter suite, replay all Stage270 saved answers, live-judge every changed answer with seed 42, then evaluate a full 500-question candidate over frozen retrieval and record paired outcomes plus retrieval-order equality.
