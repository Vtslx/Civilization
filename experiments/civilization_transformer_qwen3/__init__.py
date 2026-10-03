"""Qwen3-based real-model experiments for the civilization architecture."""

__all__ = ["Qwen3Backend", "Qwen3BackendOutput"]


def __getattr__(name: str):
    if name in __all__:
        from .backend import Qwen3Backend, Qwen3BackendOutput

        exports = {
            "Qwen3Backend": Qwen3Backend,
            "Qwen3BackendOutput": Qwen3BackendOutput,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
