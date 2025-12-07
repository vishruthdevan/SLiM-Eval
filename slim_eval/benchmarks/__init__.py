"""Benchmark modules for SLiM-Eval."""

from .accuracy import AccuracyBenchmark
from .base import BaseBenchmark
from .energy import EnergyBenchmark
from .performance import PerformanceBenchmark

__all__ = [
    "BaseBenchmark",
    "PerformanceBenchmark",
    "EnergyBenchmark",
    "AccuracyBenchmark",
]
