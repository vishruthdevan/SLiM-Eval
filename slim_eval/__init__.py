"""SLiM-Eval: Systematic LLM quantization benchmarking."""

__version__ = "0.1.0"

from .analysis import ResultsAnalyzer
from .benchmarks.accuracy import AccuracyBenchmark
from .benchmarks.base import BaseBenchmark
from .benchmarks.energy import EnergyBenchmark
from .benchmarks.performance import PerformanceBenchmark
from .evaluator import SLiMEvaluator
from .quantization import QuantizationManager

__all__ = [
    "SLiMEvaluator",
    "ResultsAnalyzer",
    "QuantizationManager",
    "BaseBenchmark",
    "PerformanceBenchmark",
    "EnergyBenchmark",
    "AccuracyBenchmark",
    "__version__",
]
