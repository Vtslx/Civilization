__all__ = [
    "Qwen3RepresentationResult",
    "collect_qwen3_hidden_states",
    "last_non_padding_pool",
    "masked_mean_pool",
    "run_qwen3_hidden_state_baseline",
]


def __getattr__(name: str):
    if name in {
        "Qwen3RepresentationResult",
        "collect_qwen3_hidden_states",
        "last_non_padding_pool",
        "masked_mean_pool",
    }:
        from .hidden_states import Qwen3RepresentationResult, collect_qwen3_hidden_states, last_non_padding_pool, masked_mean_pool

        exports = {
            "Qwen3RepresentationResult": Qwen3RepresentationResult,
            "collect_qwen3_hidden_states": collect_qwen3_hidden_states,
            "last_non_padding_pool": last_non_padding_pool,
            "masked_mean_pool": masked_mean_pool,
        }
        return exports[name]
    if name == "run_qwen3_hidden_state_baseline":
        from .pipeline import run_qwen3_hidden_state_baseline

        return run_qwen3_hidden_state_baseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
