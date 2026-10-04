"""Earlier research lines, kept for provenance and reproducibility.

These packages are not part of the production service path. They document how
the architecture arrived at the current engine and they keep their own tests:

- ``prototype``   the first executable test bench: a NumPy Transformer-like core
                  with memory, state, and rule vectors fused by a
                  CivilizationBlock.
- ``torch_line``  the PyTorch backend line: logic datasets, codebooks, ablation
                  configurations, and the training/evaluation harnesses that the
                  later stages reuse.

Nothing in :mod:`civilization.engine` depends on this package at runtime; it is
imported by design only from tests and from the stage runners that study it.
"""

from __future__ import annotations

__all__ = ["prototype", "torch_line"]
