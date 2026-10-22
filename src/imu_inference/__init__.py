"""Public API; importing contracts does not load torch or model weights."""

__version__ = "1.0.0"

from .contracts import InputBatch, PipelineConfig, capabilities

__all__ = ["InputBatch", "PipelineConfig", "capabilities", "__version__", "InferenceEngine"]


def __getattr__(name):
    if name == "InferenceEngine":
        from .engine import InferenceEngine
        return InferenceEngine
    raise AttributeError(name)
