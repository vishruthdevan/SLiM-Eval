"""Energy consumption benchmarking for SLiM-Eval."""

import logging
import threading
import time
from typing import Any, Dict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..utils import clear_cache
from .base import BaseBenchmark

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnergyBenchmark(BaseBenchmark):
    """Benchmark for measuring energy consumption."""

    def get_result_keys(self) -> list:
        """Get the list of result keys this benchmark produces."""
        return [
            "energy_kwh",
            "energy_joules",
            "duration_seconds",
            "avg_power_watts",
            "min_power_watts",
            "max_power_watts",
            "std_power_watts",
            "energy_per_query_j",
        ]

    def run(self, llm: LLM, model_name: str, precision: str) -> Dict[str, Any]:
        """Run energy consumption benchmark.

        Args:
            llm: Loaded vLLM model instance.
            model_name: Name or path of the model.
            precision: Precision mode being evaluated.

        Returns:
            Dictionary containing energy metrics.
        """
        logger.info("=" * 60)
        logger.info("ENERGY CONSUMPTION BENCHMARK (PyNVML)")
        logger.info("=" * 60)
        logger.info(
            f"Running {self.args.energy_sample_runs} inference samples with energy tracking..."
        )

        if not PYNVML_AVAILABLE:
            logger.error("PyNVML not available. Install with: pip install nvidia-ml-py")
            return {
                "energy_kwh": 0,
                "energy_joules": 0,
                "duration_seconds": 0,
                "avg_power_watts": 0,
                "min_power_watts": 0,
                "max_power_watts": 0,
                "std_power_watts": 0,
                "num_samples": 0,
                "energy_per_query_j": 0,
                "error": "PyNVML not available",
            }

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.args.max_new_tokens, top_p=1.0
        )
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main causes of climate change?",
            "Solve: If x + 5 = 12, what is x?",
            "Describe the process of photosynthesis.",
        ]

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_samples = []
            stop_monitoring = threading.Event()

            def monitor_power():
                while not stop_monitoring.is_set():
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_w = power_mw / 1000.0
                        power_samples.append(power_w)
                    except Exception as e:
                        logger.debug(f"Power sampling error: {e}")
                    time.sleep(0.1)

            monitor_thread = threading.Thread(target=monitor_power, daemon=True)
            monitor_thread.start()
            start_time = time.time()

            for i in tqdm(range(self.args.energy_sample_runs), desc="Energy tracking"):
                prompt = test_prompts[i % len(test_prompts)]
                llm.generate([prompt], sampling_params)

            end_time = time.time()
            stop_monitoring.set()
            monitor_thread.join(timeout=2.0)
            pynvml.nvmlShutdown()

            duration = end_time - start_time
            if len(power_samples) == 0:
                raise Exception("No power samples collected")

            power_array = np.array(power_samples)
            avg_power = power_array.mean()
            energy_joules = avg_power * duration
            energy_kwh = energy_joules / 3600000

            results = {
                "energy_kwh": energy_kwh,
                "energy_joules": energy_joules,
                "duration_seconds": duration,
                "avg_power_watts": avg_power,
                "min_power_watts": power_array.min(),
                "max_power_watts": power_array.max(),
                "std_power_watts": power_array.std(),
                "num_samples": self.args.energy_sample_runs,
                "energy_per_query_j": energy_joules / self.args.energy_sample_runs,
                "power_samples_collected": len(power_samples),
            }

            logger.info(
                f"Energy: {results['energy_kwh'] * 1000:.4f} Wh | Avg Power: {results['avg_power_watts']:.2f}W | "
                f"Range: {results['min_power_watts']:.1f}-{results['max_power_watts']:.1f}W"
            )

            # Clear GPU memory after energy benchmark
            clear_cache()

            return results
        except Exception as e:
            logger.error(f"Energy tracking failed: {e}", exc_info=True)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

            # Clear GPU memory after energy benchmark error
            clear_cache()

            return {
                "energy_kwh": 0,
                "energy_joules": 0,
                "duration_seconds": 0,
                "avg_power_watts": 0,
                "min_power_watts": 0,
                "max_power_watts": 0,
                "std_power_watts": 0,
                "num_samples": 0,
                "energy_per_query_j": 0,
                "error": str(e),
            }
