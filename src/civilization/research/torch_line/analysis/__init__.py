"""Analysis helpers for the PyTorch research line.

Names are resolved lazily so that importing the engine never drags in the
plotting, clustering, or projection libraries that only the analysis
workflows need.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, str] = {
    "AlignmentTrainingResult": ".alignment",
    "AnalysisResult": ".pipeline",
    "ChainStateAlignmentResult": ".chain_alignment",
    "CivilizationFusionController": ".fusion",
    "FusionDecision": ".fusion",
    "HARD_LOGIC_SCENARIOS": ".dataset",
    "InjectionTrace": ".injection",
    "LOGIC_LABELS": ".dataset",
    "LOGIC_VARIANTS": ".dataset",
    "LogicCodeInjector": ".injection",
    "LogicCodebook": ".codebook",
    "LogicSample": ".dataset",
    "LogicTokenizer": ".dataset",
    "build_hard_logic_datasets": ".dataset",
    "build_logic_codebook": ".codebook",
    "build_logic_codebook_train_test": ".codebook",
    "build_logic_dataset": ".dataset",
    "build_logic_variant_datasets": ".dataset",
    "collect_hidden_state_representations": ".hidden_states",
    "leakage_tokens_for_label": ".dataset",
    "nearest_centroid_label": ".injection",
    "run_alignment_training": ".alignment",
    "run_chain_state_alignment_training": ".chain_alignment",
    "run_codebook_generalization_matrix": ".codebook",
    "run_codebook_matrix": ".codebook",
    "run_hidden_state_analysis": ".pipeline",
    "run_single_codebook_analysis": ".codebook",
    "select_alignment_train_samples": ".alignment",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str):
    """Import the submodule that defines ``name`` on first access."""

    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)
