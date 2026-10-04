# Civilization Transformer Torch Experiment

This package is the PyTorch backend for the Route B Civilization Model test
bench. It keeps the NumPy implementation as a readable baseline and proves that
the same memory/state/rule architecture can run in a real deep-learning
framework with a minimal training loop.

## Environment

Use the repository virtual environment:

```bash
.venv/bin/python -m pytest src/civilization/research/torch_line/tests -v
```

Verified environment:

```text
Python 3.12.11
torch 2.12.0
pytest 9.0.3
MPS available on this machine
```

## What This Stage Proves

- PyTorch `MiniTransformerTorch` exports logits, hidden states, and attention.
- Memory vectors can enter the token stream and affect logits.
- State vectors and gates produce different controls for strict/creative modes.
- Rule vectors can participate in `CivilizationBlockTorch`.
- A tiny next-token dataset can train end-to-end and reduce loss.

This is still a small experimental model. It is not a real LLM and does not yet
attempt hidden-state injection into an existing open-source model.
