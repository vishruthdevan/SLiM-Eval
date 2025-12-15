"""Benchmark modules for SLiM-Eval."""

# Lazy imports to avoid dependency errors
__all__ = [
    "BaseBenchmark",
    "PerformanceBenchmark",
    "EnergyBenchmark",
    "AccuracyBenchmark",
]


def __getattr__(name):
    if name == "AccuracyBenchmark":
        from .accuracy import AccuracyBenchmark

        return AccuracyBenchmark
    elif name == "BaseBenchmark":
        from .base import BaseBenchmark

        return BaseBenchmark
    elif name == "EnergyBenchmark":
        from .energy import EnergyBenchmark

        return EnergyBenchmark
    elif name == "PerformanceBenchmark":
        from .performance import PerformanceBenchmark

        return PerformanceBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
