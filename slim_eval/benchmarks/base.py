"""Base benchmark class for SLiM-Eval."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from vllm import LLM


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    def __init__(self, args):
        """Initialize benchmark with arguments.

        Args:
            args: Arguments object containing benchmark settings.
        """
        self.args = args

    @abstractmethod
    def run(self, llm: LLM, model_name: str, precision: str) -> Dict[str, Any]:
        """Run the benchmark.

        Args:
            llm: Loaded vLLM model instance.
            model_name: Name or path of the model.
            precision: Precision mode being evaluated.

        Returns:
            Dictionary containing benchmark results.
        """
        pass

    @abstractmethod
    def get_result_keys(self) -> list:
        """Get the list of result keys this benchmark produces.

        Returns:
            List of result key names.
        """
        pass
