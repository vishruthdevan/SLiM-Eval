"""SLiM-Eval: Systematic LLM quantization benchmarking."""

__version__ = "0.1.2"

from .analysis import ResultsAnalyzer

# Only import these when needed to avoid dependency errors
__all__ = [
    "ResultsAnalyzer",
    "__version__",
]
