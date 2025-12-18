import asyncio
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class BenchmarkTask:
    """Represents a single benchmark task"""
    task_id: str
    model_name: str
    precision: str
    benchmark_type: str
    device_id: int
    config: Dict[str, Any]


@dataclass
class BenchmarkTaskResult:
    """Result of a benchmark task"""
    task_id: str
    model_name: str
    precision: str
    benchmark_type: str
    device_id: int
    success: bool
    metrics: Dict[str, Any]
    error: Optional[str] = None
    duration_s: float = 0.0
    timestamp: str = ""


class DeviceManager:
    """Manages available compute devices"""
    
    def __init__(self):
        self.available_devices = self._discover_devices()
        print(f"Discovered {len(self.available_devices)} devices: {self.available_devices}")
    
    def _discover_devices(self) -> List[Dict]:
        """Discover available compute devices"""
        devices = []
        
        # Check for CUDA GPUs
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "type": "cuda",
                        "id": i,
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
        except ImportError:
            pass
        
        # Add CPU as fallback
        devices.append({
            "type": "cpu",
            "id": -1,
            "name": "CPU",
            "cores": mp.cpu_count(),
            "memory_gb": self._get_system_memory_gb()
        })
        
        return devices
    
    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 0.0
    
    def get_device_by_id(self, device_id: int) -> Optional[Dict]:
        """Get device info by ID"""
        for device in self.available_devices:
            if device["id"] == device_id:
                return device
        return None
    
    def get_cuda_devices(self) -> List[Dict]:
        """Get only CUDA devices"""
        return [d for d in self.available_devices if d["type"] == "cuda"]
    
    def get_cpu_device(self) -> Optional[Dict]:
        """Get CPU device"""
        for device in self.available_devices:
            if device["type"] == "cpu":
                return device
        return None


def run_latency_benchmark_on_device(task: BenchmarkTask) -> BenchmarkTaskResult:
    """Run latency benchmark on specific device (runs in separate process)"""
    start_time = time.time()
    
    try:
        # Set device environment
        if task.device_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task.device_id)
        
        # Import here to avoid issues with multiprocessing
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Configure quantization
        quant_config = None
        if task.precision == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif task.precision == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(task.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            task.model_name,
            quantization_config=quant_config,
            device_map="auto" if task.device_id >= 0 else "cpu",
            torch_dtype=torch.float16 if task.precision == "fp16" else "auto"
        )
        
        # Benchmark configuration
        num_runs = task.config.get("num_runs", 20)
        warmup_runs = task.config.get("warmup_runs", 3)
        max_tokens = task.config.get("max_tokens", 32)
        test_prompt = task.config.get("test_prompt", "Explain AI briefly.")
        
        # Warmup
        for _ in range(warmup_runs):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if task.device_id >= 0:
                inputs = {k: v.to(f"cuda:{task.device_id}") for k, v in inputs.items()}
            
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Actual benchmark
        latencies = []
        for _ in range(num_runs):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if task.device_id >= 0:
                inputs = {k: v.to(f"cuda:{task.device_id}") for k, v in inputs.items()}
            
            iter_start = time.time()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_tokens)
            iter_end = time.time()
            
            latencies.append(iter_end - iter_start)
        
        # Calculate metrics
        metrics = {
            "mean_latency_s": np.mean(latencies),
            "median_latency_s": np.median(latencies),
            "std_latency_s": np.std(latencies),
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "num_runs": len(latencies)
        }
        
        # Memory metrics
        if task.device_id >= 0 and torch.cuda.is_available():
            metrics["gpu_memory_mb"] = torch.cuda.memory_allocated(task.device_id) / (1024**2)
            metrics["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated(task.device_id) / (1024**2)
        
        return BenchmarkTaskResult(
            task_id=task.task_id,
            model_name=task.model_name,
            precision=task.precision,
            benchmark_type=task.benchmark_type,
            device_id=task.device_id,
            success=True,
            metrics=metrics,
            duration_s=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return BenchmarkTaskResult(
            task_id=task.task_id,
            model_name=task.model_name,
            precision=task.precision,
            benchmark_type=task.benchmark_type,
            device_id=task.device_id,
            success=False,
            metrics={},
            error=str(e),
            duration_s=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )


def run_memory_benchmark_on_device(task: BenchmarkTask) -> BenchmarkTaskResult:
    """Run memory benchmark on specific device"""
    start_time = time.time()
    
    try:
        # Set device environment
        if task.device_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task.device_id)
        
        import torch
        import psutil
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Baseline memory
        baseline_ram_mb = psutil.virtual_memory().used / (1024**2)
        baseline_gpu_mb = 0
        if task.device_id >= 0 and torch.cuda.is_available():
            baseline_gpu_mb = torch.cuda.memory_allocated(task.device_id) / (1024**2)
        
        # Configure and load model (similar to latency benchmark)
        quant_config = None
        if task.precision == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif task.precision == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        tokenizer = AutoTokenizer.from_pretrained(task.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            task.model_name,
            quantization_config=quant_config,
            device_map="auto" if task.device_id >= 0 else "cpu",
            torch_dtype=torch.float16 if task.precision == "fp16" else "auto"
        )
        
        # Reset peak memory stats
        if task.device_id >= 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(task.device_id)
        
        # Run some inferences to measure peak memory
        test_prompt = task.config.get("test_prompt", "Explain AI briefly.")
        for _ in range(5):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if task.device_id >= 0:
                inputs = {k: v.to(f"cuda:{task.device_id}") for k, v in inputs.items()}
            
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=64)
        
        # Measure final memory
        final_ram_mb = psutil.virtual_memory().used / (1024**2)
        final_gpu_mb = 0
        peak_gpu_mb = 0
        
        if task.device_id >= 0 and torch.cuda.is_available():
            final_gpu_mb = torch.cuda.memory_allocated(task.device_id) / (1024**2)
            peak_gpu_mb = torch.cuda.max_memory_allocated(task.device_id) / (1024**2)
        
        metrics = {
            "baseline_ram_mb": baseline_ram_mb,
            "final_ram_mb": final_ram_mb,
            "ram_increase_mb": final_ram_mb - baseline_ram_mb,
            "baseline_gpu_mb": baseline_gpu_mb,
            "final_gpu_mb": final_gpu_mb,
            "peak_gpu_mb": peak_gpu_mb,
            "gpu_increase_mb": final_gpu_mb - baseline_gpu_mb
        }
        
        return BenchmarkTaskResult(
            task_id=task.task_id,
            model_name=task.model_name,
            precision=task.precision,
            benchmark_type=task.benchmark_type,
            device_id=task.device_id,
            success=True,
            metrics=metrics,
            duration_s=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return BenchmarkTaskResult(
            task_id=task.task_id,
            model_name=task.model_name,
            precision=task.precision,
            benchmark_type=task.benchmark_type,
            device_id=task.device_id,
            success=False,
            metrics={},
            error=str(e),
            duration_s=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )


class DistributedBenchmarkManager:
    """Manages distributed benchmarking across multiple devices"""
    
    def __init__(self, output_dir: str = "distributed_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device_manager = DeviceManager()
        self.task_queue = queue.Queue()
        self.results = []
        self.active_tasks = {}
        
        # Benchmark function mapping
        self.benchmark_functions = {
            "latency": run_latency_benchmark_on_device,
            "memory": run_memory_benchmark_on_device,
        }
        
        print(f"Distributed Benchmark Manager initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Available devices: {len(self.device_manager.available_devices)}")
    
    def create_benchmark_tasks(self, models: List[str], precisions: List[str], 
                             benchmark_types: List[str], config: Dict = None) -> List[BenchmarkTask]:
        """Create all benchmark tasks"""
        tasks = []
        task_id = 0
        
        for model in models:
            for precision in precisions:
                for benchmark_type in benchmark_types:
                    # Create tasks for each available device
                    for device in self.device_manager.available_devices:
                        task = BenchmarkTask(
                            task_id=f"task_{task_id:04d}",
                            model_name=model,
                            precision=precision,
                            benchmark_type=benchmark_type,
                            device_id=device["id"],
                            config=config or {}
                        )
                        tasks.append(task)
                        task_id += 1
        
        return tasks
    
    def run_distributed_benchmarks(self, tasks: List[BenchmarkTask], max_workers: int = None) -> List[BenchmarkTaskResult]:
        """Run benchmarks in parallel across devices"""
        if max_workers is None:
            max_workers = len(self.device_manager.available_devices)
        
        print(f"\nRunning {len(tasks)} tasks across {max_workers} workers...")
        
        results = []
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                benchmark_func = self.benchmark_functions.get(task.benchmark_type)
                if benchmark_func:
                    future = executor.submit(benchmark_func, task)
                    future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        print(f"  ✓ {task.task_id}: {task.benchmark_type} on device {task.device_id} "
                              f"({task.model_name}, {task.precision}) - {result.duration_s:.1f}s")
                    else:
                        print(f"  ✗ {task.task_id}: Failed - {result.error}")
                    
                    # Save result immediately
                    self._save_result_immediately(result)
                    
                except Exception as e:
                    print(f"  ✗ {task.task_id}: Exception - {e}")
        
        self.results.extend(results)
        return results
    
    def run_sequential_benchmarks(self, tasks: List[BenchmarkTask]) -> List[BenchmarkTaskResult]:
        """Run benchmarks sequentially (for debugging)"""
        print(f"\nRunning {len(tasks)} tasks sequentially...")
        
        results = []
        
        for i, task in enumerate(tasks):
            print(f"Running task {i+1}/{len(tasks)}: {task.task_id}")
            
            benchmark_func = self.benchmark_functions.get(task.benchmark_type)
            if benchmark_func:
                try:
                    result = benchmark_func(task)
                    results.append(result)
                    
                    if result.success:
                        print(f"  ✓ Completed in {result.duration_s:.1f}s")
                    else:
                        print(f"  ✗ Failed: {result.error}")
                    
                    self._save_result_immediately(result)
                    
                except Exception as e:
                    print(f"  ✗ Exception: {e}")
            else:
                print(f"  ✗ Unknown benchmark type: {task.benchmark_type}")
        
        self.results.extend(results)
        return results
    
    def _save_result_immediately(self, result: BenchmarkTaskResult):
        """Save individual result immediately"""
        result_file = self.output_dir / f"{result.task_id}_result.json"
        
        result_dict = {
            "task_id": result.task_id,
            "model_name": result.model_name,
            "precision": result.precision,
            "benchmark_type": result.benchmark_type,
            "device_id": result.device_id,
            "success": result.success,
            "metrics": result.metrics,
            "error": result.error,
            "duration_s": result.duration_s,
            "timestamp": result.timestamp
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def analyze_device_performance(self) -> Dict:
        """Analyze performance across different devices"""
        if not self.results:
            return {}
        
        device_analysis = {}
        
        for result in self.results:
            if not result.success:
                continue
            
            device_id = result.device_id
            if device_id not in device_analysis:
                device_info = self.device_manager.get_device_by_id(device_id)
                device_analysis[device_id] = {
                    "device_info": device_info,
                    "results": [],
                    "avg_latency": 0,
                    "avg_memory": 0,
                    "total_tasks": 0,
                    "successful_tasks": 0
                }
            
            device_analysis[device_id]["results"].append(result)
            device_analysis[device_id]["total_tasks"] += 1
            device_analysis[device_id]["successful_tasks"] += 1
        
        # Calculate averages
        for device_id, analysis in device_analysis.items():
            latencies = []
            memories = []
            
            for result in analysis["results"]:
                if "mean_latency_s" in result.metrics:
                    latencies.append(result.metrics["mean_latency_s"])
                if "peak_gpu_mb" in result.metrics:
                    memories.append(result.metrics["peak_gpu_mb"])
            
            analysis["avg_latency"] = np.mean(latencies) if latencies else 0
            analysis["avg_memory"] = np.mean(memories) if memories else 0
        
        return device_analysis
    
    def generate_distributed_report(self):
        """Generate comprehensive distributed benchmark report"""
        print(f"\n{'='*60}")
        print("DISTRIBUTED BENCHMARK REPORT")
        print(f"{'='*60}")
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Overall statistics
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        print(f"\nOverall Statistics:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {successful_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Success rate: {successful_tasks/total_tasks*100:.1f}%")
        
        # Device performance analysis
        device_analysis = self.analyze_device_performance()
        
        print(f"\nDevice Performance Analysis:")
        print(f"{'Device':<15} {'Type':<8} {'Avg Latency (s)':<15} {'Avg Memory (MB)':<15} {'Tasks':<8}")
        print("-" * 70)
        
        for device_id, analysis in device_analysis.items():
            device_info = analysis["device_info"]
            device_name = device_info["name"][:12] if device_info else "Unknown"
            device_type = device_info["type"] if device_info else "Unknown"
            
            print(f"{device_name:<15} {device_type:<8} {analysis['avg_latency']:<15.3f} "
                  f"{analysis['avg_memory']:<15.1f} {analysis['successful_tasks']:<8}")
        
        # Model/Precision breakdown
        model_precision_stats = {}
        for result in self.results:
            if not result.success:
                continue
            
            key = f"{result.model_name}_{result.precision}"
            if key not in model_precision_stats:
                model_precision_stats[key] = {"latency": [], "memory": []}
            
            if "mean_latency_s" in result.metrics:
                model_precision_stats[key]["latency"].append(result.metrics["mean_latency_s"])
            if "peak_gpu_mb" in result.metrics:
                model_precision_stats[key]["memory"].append(result.metrics["peak_gpu_mb"])
        
        print(f"\nModel/Precision Performance:")
        print(f"{'Model/Precision':<30} {'Avg Latency (s)':<15} {'Avg Memory (MB)':<15}")
        print("-" * 65)
        
        for key, stats in model_precision_stats.items():
            avg_latency = np.mean(stats["latency"]) if stats["latency"] else 0
            avg_memory = np.mean(stats["memory"]) if stats["memory"] else 0
            print(f"{key:<30} {avg_latency:<15.3f} {avg_memory:<15.1f}")
        
        # Save detailed analysis
        analysis_file = self.output_dir / "distributed_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                "overall_stats": {
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                    "success_rate": successful_tasks/total_tasks if total_tasks > 0 else 0
                },
                "device_analysis": device_analysis,
                "model_precision_stats": {
                    key: {
                        "avg_latency": np.mean(stats["latency"]) if stats["latency"] else 0,
                        "avg_memory": np.mean(stats["memory"]) if stats["memory"] else 0,
                        "num_samples": len(stats["latency"])
                    }
                    for key, stats in model_precision_stats.items()
                }
            }, f, indent=2, default=str)
        
        print(f"\nDetailed analysis saved to: {analysis_file}")


def main():
    """Main distributed benchmark runner"""
    manager = DistributedBenchmarkManager()
    
    # Test configuration
    test_models = [
        "microsoft/DialoGPT-small",  # Small model for testing
        # "microsoft/Phi-3-mini-4k-instruct",  # Uncomment for larger tests
    ]
    test_precisions = ["fp16", "int8"]
    test_benchmarks = ["latency", "memory"]
    
    benchmark_config = {
        "num_runs": 10,
        "warmup_runs": 2,
        "max_tokens": 32,
        "test_prompt": "Explain AI in one sentence."
    }
    
    # Create tasks
    tasks = manager.create_benchmark_tasks(
        models=test_models,
        precisions=test_precisions,
        benchmark_types=test_benchmarks,
        config=benchmark_config
    )
    
    print(f"Created {len(tasks)} benchmark tasks")
    
    # Run benchmarks
    if len(manager.device_manager.get_cuda_devices()) > 1:
        print("Multiple GPUs detected, running distributed benchmarks...")
        results = manager.run_distributed_benchmarks(tasks, max_workers=4)
    else:
        print("Single device detected, running sequential benchmarks...")
        results = manager.run_sequential_benchmarks(tasks)
    
    # Generate report
    manager.generate_distributed_report()
    
    print(f"\n{'='*60}")
    print("Distributed Benchmark Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()