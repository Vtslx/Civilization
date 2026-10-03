# Civilization Transformer Experiment

This directory contains the first executable test bench for the Route B
Civilization Model architecture.

The goal is not to build a production LLM. The goal is to prove, with small
deterministic experiments, that memory, state, and rule signals can move from
external configuration into the model computation path.

## Current Backend

The first implementation uses NumPy because the current local environment has
NumPy available but does not have PyTorch, pytest, matplotlib, or sklearn
installed. The module boundaries intentionally mirror a future PyTorch version:

```text
model/   Transformer-like neural core and CivilizationBlock
memory/  MemoryItem and MemoryEncoder
state/   StateConfig, StateEncoder, and StateGate
rules/   RuleItem, RuleEngine, and rule vector encoding
tests/   Multi-round comparison tests
```

## Phase Mapping

- Phase 0: This README documents Transformer data flow and execution constraints.
- Phase 1: `model/transformer.py` implements a minimal Transformer-like core.
- Phase 2: `memory/memory.py` implements memory vectors and memory token input.
- Phase 3: `state/state.py` implements state vectors and state gates.
- Phase 4: `rules/rules.py` implements hard, soft, and conflict rules.
- Phase 5: `model/civilization_block.py` fuses hidden states with memory, state, and rule vectors.
- Phase 6: tests inspect hidden states as a first semantic-observation hook.
- Phase 7: the first end-to-end results are summarized in the repository root README.

## Transformer Data Flow

```text
token ids
-> token embedding
-> positional embedding
-> Transformer blocks
   -> attention
   -> residual + normalization
   -> MLP
   -> residual + normalization
-> hidden states
-> LM Head logits
```

Tokenizer is not the Transformer. A tokenizer converts human text into token
ids. The Transformer consumes token ids and produces contextual hidden states.
In this experiment, tests feed token ids directly so the neural computation can
be inspected without a tokenizer dependency.

## Hidden States Value

Hidden states are the observable internal semantic surface for Route B. They are
where memory vectors, state gates, and rule vectors must eventually interact if
this architecture is to move beyond prompt-level augmentation.

## Run Tests

```bash
python3 -m unittest discover -s experiments/civilization_transformer/tests -v
```
