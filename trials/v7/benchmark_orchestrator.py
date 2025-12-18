import asyncio
import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm import tqdm


class BenchmarkStatus(Enum):
    """Benchmark execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics"""
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_std: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    energy_joules: float = 0.0
    energy_kwh: float = 0.0
    power_avg_watts: float = 0.0
    accuracy_score: float = 0.0
    model_size_mb: float = 0.0
    inference_count: int = 0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkTask:
    """Comprehensive benchmark task definition"""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    model_name: str = ""
    precision: str = "fp16"
    benchmark_type: str = "latency"
    priority: Priority = Priority.NORMAL
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    metrics: Optional[BenchmarkMetrics] = None
    error_message: Optional[str] = None
    device_id: Optional[int] = None
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'scheduled_at', 'started_at', 'completed_at']:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        return result


class BenchmarkExecutor(ABC):
    """Abstract base class for benchmark executors"""
    
    def __init__(self, executor_id: str):
        self.executor_id = executor_id
        self.is_busy = False
        self.current_task = None
        
    @abstractmethod
    async def execute(self, task: BenchmarkTask) -> BenchmarkMetrics:
        """Execute a benchmark task"""
        pass
    
    @abstractmethod
    def can_handle(self, task: BenchmarkTask) -> bool:
        """Check if this executor can handle the task"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass


class LatencyBenchmarkExecutor(BenchmarkExecutor):
    """Latency benchmark executor"""
    
    def __init__(self, executor_id: str, device_id: int = 0):
        super().__init__(executor_id)
        self.device_id = device_id
        
    def can_handle(self, task: BenchmarkTask) -> bool:
        return task.benchmark_type == "latency"
    
    async def execute(self, task: BenchmarkTask) -> BenchmarkMetrics:
        """Execute latency benchmark"""
        self.is_busy = True
        self.current_task = task
        
        try:
            # Simulate latency benchmark execution
            await asyncio.sleep(0.1)  # Simulate setup time
            
            num_runs = task.config.get("num_runs", 100)
            max_tokens = task.config.get("max_tokens", 32)
            
            # Simulate benchmark runs
            latencies = []
            for i in range(num_runs):
                # Simulate inference latency (varies by model and precision)
                base_latency = self._get_base_latency(task.model_name, task.precision)
                noise = np.random.normal(0, base_latency * 0.1)
                latency = max(0.001, base_latency + noise)
                latencies.append(latency)
                
                if i % 10 == 0:
                    await asyncio.sleep(0.01)  # Yield control
            
            lat_array = np.array(latencies)
            
            metrics = BenchmarkMetrics(
                latency_mean=lat_array.mean(),
                latency_p50=np.percentile(lat_array, 50),
                latency_p95=np.percentile(lat_array, 95),
                latency_p99=np.percentile(lat_array, 99),
                latency_std=lat_array.std(),
                throughput_tokens_per_sec=max_tokens / lat_array.mean(),
                inference_count=num_runs
            )
            
            return metrics
            
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _get_base_latency(self, model_name: str, precision: str) -> float:
        """Get base latency for model/precision combination"""
        # Simulate different model sizes and precisions
        model_size_factor = 1.0
        if "small" in model_name.lower():
            model_size_factor = 0.5
        elif "large" in model_name.lower():
            model_size_factor = 2.0
        elif "xl" in model_name.lower():
            model_size_factor = 4.0
        
        precision_factor = {"fp16": 1.0, "int8": 0.7, "int4": 0.5}.get(precision, 1.0)
        
        return 0.1 * model_size_factor / precision_factor
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_busy = False
        self.current_task = None


class MemoryBenchmarkExecutor(BenchmarkExecutor):
    """Memory benchmark executor"""
    
    def __init__(self, executor_id: str, device_id: int = 0):
        super().__init__(executor_id)
        self.device_id = device_id
        
    def can_handle(self, task: BenchmarkTask) -> bool:
        return task.benchmark_type == "memory"
    
    async def execute(self, task: BenchmarkTask) -> BenchmarkMetrics:
        """Execute memory benchmark"""
        self.is_busy = True
        self.current_task = task
        
        try:
            await asyncio.sleep(0.05)  # Simulate setup
            
            # Simulate memory usage based on model and precision
            base_memory = self._get_base_memory(task.model_name, task.precision)
            peak_memory = base_memory * 1.2  # Peak is 20% higher
            
            metrics = BenchmarkMetrics(
                memory_peak_mb=peak_memory,
                memory_avg_mb=base_memory,
                model_size_mb=base_memory * 0.8
            )
            
            return metrics
            
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _get_base_memory(self, model_name: str, precision: str) -> float:
        """Get base memory usage for model/precision"""
        # Simulate memory usage
        base_size = 1000  # MB
        
        if "small" in model_name.lower():
            base_size = 500
        elif "large" in model_name.lower():
            base_size = 2000
        elif "xl" in model_name.lower():
            base_size = 4000
        
        precision_factor = {"fp16": 1.0, "int8": 0.5, "int4": 0.25}.get(precision, 1.0)
        
        return base_size * precision_factor
    
    async def cleanup(self):
        self.is_busy = False
        self.current_task = None


class EnergyBenchmarkExecutor(BenchmarkExecutor):
    """Energy benchmark executor"""
    
    def __init__(self, executor_id: str, device_id: int = 0):
        super().__init__(executor_id)
        self.device_id = device_id
        
    def can_handle(self, task: BenchmarkTask) -> bool:
        return task.benchmark_type == "energy"
    
    async def execute(self, task: BenchmarkTask) -> BenchmarkMetrics:
        """Execute energy benchmark"""
        self.is_busy = True
        self.current_task = task
        
        try:
            duration = task.config.get("duration_seconds", 10)
            await asyncio.sleep(duration * 0.01)  # Simulate measurement time
            
            # Simulate energy measurements
            base_power = self._get_base_power(task.model_name, task.precision)
            energy_joules = base_power * duration
            
            metrics = BenchmarkMetrics(
                energy_joules=energy_joules,
                energy_kwh=energy_joules / 3600000,
                power_avg_watts=base_power
            )
            
            return metrics
            
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _get_base_power(self, model_name: str, precision: str) -> float:
        """Get base power consumption"""
        base_power = 150  # Watts
        
        if "small" in model_name.lower():
            base_power = 100
        elif "large" in model_name.lower():
            base_power = 250
        elif "xl" in model_name.lower():
            base_power = 400
        
        precision_factor = {"fp16": 1.0, "int8": 0.8, "int4": 0.6}.get(precision, 1.0)
        
        return base_power * precision_factor
    
    async def cleanup(self):
        self.is_busy = False
        self.current_task = None


class AccuracyBenchmarkExecutor(BenchmarkExecutor):
    """Accuracy benchmark executor"""
    
    def __init__(self, executor_id: str):
        super().__init__(executor_id)
        
    def can_handle(self, task: BenchmarkTask) -> bool:
        return task.benchmark_type == "accuracy"
    
    async def execute(self, task: BenchmarkTask) -> BenchmarkMetrics:
        """Execute accuracy benchmark"""
        self.is_busy = True
        self.current_task = task
        
        try:
            num_samples = task.config.get("num_samples", 1000)
            await asyncio.sleep(num_samples * 0.001)  # Simulate evaluation time
            
            # Simulate accuracy based on model and precision
            base_accuracy = self._get_base_accuracy(task.model_name, task.precision)
            
            metrics = BenchmarkMetrics(
                accuracy_score=base_accuracy,
                inference_count=num_samples
            )
            
            return metrics
            
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _get_base_accuracy(self, model_name: str, precision: str) -> float:
        """Get base accuracy for model/precision"""
        base_accuracy = 0.75
        
        if "small" in model_name.lower():
            base_accuracy = 0.65
        elif "large" in model_name.lower():
            base_accuracy = 0.85
        elif "xl" in model_name.lower():
            base_accuracy = 0.90
        
        # Precision degradation
        precision_penalty = {"fp16": 0.0, "int8": 0.02, "int4": 0.05}.get(precision, 0.0)
        
        return max(0.0, base_accuracy - precision_penalty)
    
    async def cleanup(self):
        self.is_busy = False
        self.current_task = None


class TaskScheduler:
    """Advanced task scheduler with priority queues and dependencies"""
    
    def __init__(self):
        self.task_queues = {
            Priority.CRITICAL: queue.PriorityQueue(),
            Priority.HIGH: queue.PriorityQueue(),
            Priority.NORMAL: queue.PriorityQueue(),
            Priority.LOW: queue.PriorityQueue()
        }
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_dependencies = {}
        self.lock = threading.Lock()
        
    def add_task(self, task: BenchmarkTask):
        """Add task to scheduler"""
        with self.lock:
            self.pending_tasks[task.task_id] = task
            
            # Check if dependencies are satisfied
            if self._dependencies_satisfied(task):
                self._queue_task(task)
            else:
                # Store for later when dependencies are satisfied
                self.task_dependencies[task.task_id] = task.dependencies.copy()
    
    def _dependencies_satisfied(self, task: BenchmarkTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _queue_task(self, task: BenchmarkTask):
        """Queue task based on priority"""
        priority_value = task.priority.value
        timestamp = time.time()
        # Use negative priority for max-heap behavior (higher priority first)
        self.task_queues[task.priority].put((-priority_value, timestamp, task))
    
    def get_next_task(self) -> Optional[BenchmarkTask]:
        """Get next task to execute"""
        with self.lock:
            # Check queues in priority order
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                if not self.task_queues[priority].empty():
                    _, _, task = self.task_queues[priority].get()
                    task.scheduled_at = datetime.now()
                    return task
            return None
    
    def mark_completed(self, task: BenchmarkTask):
        """Mark task as completed and check for newly available tasks"""
        with self.lock:
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            
            # Check if any pending tasks can now be queued
            newly_available = []
            for task_id, dependencies in list(self.task_dependencies.items()):
                if task.task_id in dependencies:
                    dependencies.remove(task.task_id)
                    if not dependencies:
                        # All dependencies satisfied
                        pending_task = self.pending_tasks.get(task_id)
                        if pending_task:
                            newly_available.append(pending_task)
                        del self.task_dependencies[task_id]
            
            # Queue newly available tasks
            for available_task in newly_available:
                self._queue_task(available_task)
    
    def mark_failed(self, task: BenchmarkTask):
        """Mark task as failed"""
        with self.lock:
            self.failed_tasks[task.task_id] = task
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
    
    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        with self.lock:
            total_queued = sum(q.qsize() for q in self.task_queues.values())
            return {
                "pending": len(self.pending_tasks),
                "queued": total_queued,
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "waiting_for_dependencies": len(self.task_dependencies)
            }


class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.resource_history = []
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.is_monitoring:
            try:
                resources = self._collect_resources()
                with self.lock:
                    self.resource_history.append(resources)
                    # Keep only last 1000 entries
                    if len(self.resource_history) > 1000:
                        self.resource_history = self.resource_history[-1000:]
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
    
    def _collect_resources(self) -> Dict:
        """Collect current resource usage"""
        timestamp = datetime.now()
        
        # Simulate resource collection
        cpu_percent = np.random.uniform(20, 80)
        memory_percent = np.random.uniform(30, 70)
        gpu_memory_mb = np.random.uniform(1000, 8000)
        gpu_utilization = np.random.uniform(10, 90)
        
        return {
            "timestamp": timestamp.isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_utilization": gpu_utilization
        }
    
    def get_current_resources(self) -> Dict:
        """Get current resource usage"""
        return self._collect_resources()
    
    def get_resource_history(self, minutes: int = 10) -> List[Dict]:
        """Get resource history for specified minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            filtered_history = []
            for entry in self.resource_history:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time >= cutoff_time:
                    filtered_history.append(entry)
            return filtered_history


class BenchmarkOrchestrator:
    """Main orchestrator for enterprise-grade benchmarking"""
    
    def __init__(self, output_dir: str = "benchmark_results", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_workers = max_workers
        self.scheduler = TaskScheduler()
        self.resource_monitor = ResourceMonitor()
        self.executors = []
        self.worker_pool = None
        self.is_running = False
        self.orchestrator_thread = None
        
        # Statistics
        self.start_time = None
        self.total_tasks_processed = 0
        self.total_execution_time = 0.0
        
        # Setup logging
        self._setup_logging()
        
        # Initialize executors
        self._initialize_executors()
        
        logging.info(f"Benchmark Orchestrator initialized")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Max workers: {self.max_workers}")
        logging.info(f"Executors: {len(self.executors)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "orchestrator.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_executors(self):
        """Initialize benchmark executors"""
        # Create multiple executors for different benchmark types
        for i in range(2):  # 2 latency executors
            self.executors.append(LatencyBenchmarkExecutor(f"latency_{i}", device_id=i % 2))
        
        for i in range(2):  # 2 memory executors
            self.executors.append(MemoryBenchmarkExecutor(f"memory_{i}", device_id=i % 2))
        
        self.executors.append(EnergyBenchmarkExecutor("energy_0", device_id=0))
        self.executors.append(AccuracyBenchmarkExecutor("accuracy_0"))
        
        logging.info(f"Initialized {len(self.executors)} executors")
    
    async def start_orchestration(self):
        """Start the orchestration process"""
        if self.is_running:
            logging.warning("Orchestrator already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Start worker pool
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logging.info("Orchestration started")
        
        # Main orchestration loop
        try:
            await self._orchestration_loop()
        finally:
            await self.stop_orchestration()
    
    async def stop_orchestration(self):
        """Stop the orchestration process"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Shutdown worker pool
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        # Cleanup executors
        for executor in self.executors:
            await executor.cleanup()
        
        logging.info("Orchestration stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.is_running:
            try:
                # Get next task from scheduler
                task = self.scheduler.get_next_task()
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Find available executor
                executor = self._find_available_executor(task)
                
                if executor is None:
                    # No available executor, put task back and wait
                    # Note: In a real implementation, we'd need to re-queue the task
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                asyncio.create_task(self._execute_task(task, executor))
                
            except Exception as e:
                logging.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(1.0)
    
    def _find_available_executor(self, task: BenchmarkTask) -> Optional[BenchmarkExecutor]:
        """Find available executor for task"""
        for executor in self.executors:
            if not executor.is_busy and executor.can_handle(task):
                return executor
        return None
    
    async def _execute_task(self, task: BenchmarkTask, executor: BenchmarkExecutor):
        """Execute a single task"""
        task.status = BenchmarkStatus.RUNNING
        task.started_at = datetime.now()
        task.worker_id = executor.executor_id
        
        logging.info(f"Executing task {task.task_id} on {executor.executor_id}")
        
        try:
            # Execute with timeout
            metrics = await asyncio.wait_for(
                executor.execute(task),
                timeout=task.timeout_seconds
            )
            
            task.metrics = metrics
            task.status = BenchmarkStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Mark as completed in scheduler
            self.scheduler.mark_completed(task)
            
            # Save results
            await self._save_task_result(task)
            
            # Update statistics
            self.total_tasks_processed += 1
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.total_execution_time += execution_time
            
            logging.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.status = BenchmarkStatus.TIMEOUT
            task.error_message = f"Task timed out after {task.timeout_seconds}s"
            task.completed_at = datetime.now()
            
            self.scheduler.mark_failed(task)
            await self._save_task_result(task)
            
            logging.error(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.status = BenchmarkStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            self.scheduler.mark_failed(task)
            await self._save_task_result(task)
            
            logging.error(f"Task {task.task_id} failed: {e}")
    
    async def _save_task_result(self, task: BenchmarkTask):
        """Save task result to file"""
        result_file = self.output_dir / f"task_{task.task_id}.json"
        
        with open(result_file, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
    
    def submit_task(self, task: BenchmarkTask):
        """Submit a task for execution"""
        self.scheduler.add_task(task)
        logging.info(f"Task {task.task_id} submitted")
    
    def submit_benchmark_suite(self, models: List[str], precisions: List[str], 
                             benchmark_types: List[str], config: Dict = None) -> List[str]:
        """Submit a complete benchmark suite"""
        task_ids = []
        config = config or {}
        
        for model in models:
            for precision in precisions:
                for benchmark_type in benchmark_types:
                    task = BenchmarkTask(
                        model_name=model,
                        precision=precision,
                        benchmark_type=benchmark_type,
                        config=config.get(benchmark_type, {}),
                        priority=Priority.NORMAL
                    )
                    
                    self.submit_task(task)
                    task_ids.append(task.task_id)
        
        logging.info(f"Submitted benchmark suite: {len(task_ids)} tasks")
        return task_ids
    
    def get_orchestrator_status(self) -> Dict:
        """Get current orchestrator status"""
        scheduler_stats = self.scheduler.get_statistics()
        current_resources = self.resource_monitor.get_current_resources()
        
        executor_status = {}
        for executor in self.executors:
            executor_status[executor.executor_id] = {
                "busy": executor.is_busy,
                "current_task": executor.current_task.task_id if executor.current_task else None
            }
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        avg_execution_time = (self.total_execution_time / self.total_tasks_processed 
                            if self.total_tasks_processed > 0 else 0)
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "total_tasks_processed": self.total_tasks_processed,
            "avg_execution_time_seconds": avg_execution_time,
            "scheduler": scheduler_stats,
            "executors": executor_status,
            "resources": current_resources
        }
    
    async def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        logging.info("Generating comprehensive report...")
        
        # Collect all task results
        task_files = list(self.output_dir.glob("task_*.json"))
        all_tasks = []
        
        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                all_tasks.append(task_data)
        
        # Analyze results
        report = {
            "summary": self._analyze_summary(all_tasks),
            "performance_analysis": self._analyze_performance(all_tasks),
            "resource_analysis": self._analyze_resources(),
            "model_comparison": self._analyze_models(all_tasks),
            "precision_comparison": self._analyze_precisions(all_tasks),
            "benchmark_type_analysis": self._analyze_benchmark_types(all_tasks),
            "failure_analysis": self._analyze_failures(all_tasks),
            "recommendations": self._generate_recommendations(all_tasks)
        }
        
        # Save report
        report_file = self.output_dir / "comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Comprehensive report saved to {report_file}")
        return report
    
    def _analyze_summary(self, tasks: List[Dict]) -> Dict:
        """Analyze summary statistics"""
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t["status"] == "completed"])
        failed_tasks = len([t for t in tasks if t["status"] == "failed"])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": (self.total_execution_time / completed_tasks 
                                 if completed_tasks > 0 else 0)
        }
    
    def _analyze_performance(self, tasks: List[Dict]) -> Dict:
        """Analyze performance metrics"""
        completed_tasks = [t for t in tasks if t["status"] == "completed" and t["metrics"]]
        
        if not completed_tasks:
            return {}
        
        # Collect metrics
        latencies = []
        throughputs = []
        memory_usage = []
        energy_consumption = []
        
        for task in completed_tasks:
            metrics = task["metrics"]
            if metrics.get("latency_mean", 0) > 0:
                latencies.append(metrics["latency_mean"])
            if metrics.get("throughput_tokens_per_sec", 0) > 0:
                throughputs.append(metrics["throughput_tokens_per_sec"])
            if metrics.get("memory_peak_mb", 0) > 0:
                memory_usage.append(metrics["memory_peak_mb"])
            if metrics.get("energy_joules", 0) > 0:
                energy_consumption.append(metrics["energy_joules"])
        
        analysis = {}
        
        if latencies:
            analysis["latency"] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies)
            }
        
        if throughputs:
            analysis["throughput"] = {
                "mean": np.mean(throughputs),
                "median": np.median(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs)
            }
        
        if memory_usage:
            analysis["memory"] = {
                "mean": np.mean(memory_usage),
                "median": np.median(memory_usage),
                "std": np.std(memory_usage),
                "min": np.min(memory_usage),
                "max": np.max(memory_usage)
            }
        
        if energy_consumption:
            analysis["energy"] = {
                "mean": np.mean(energy_consumption),
                "median": np.median(energy_consumption),
                "std": np.std(energy_consumption),
                "total": np.sum(energy_consumption)
            }
        
        return analysis
    
    def _analyze_resources(self) -> Dict:
        """Analyze resource usage"""
        resource_history = self.resource_monitor.get_resource_history(minutes=60)
        
        if not resource_history:
            return {}
        
        cpu_usage = [r["cpu_percent"] for r in resource_history]
        memory_usage = [r["memory_percent"] for r in resource_history]
        gpu_memory = [r["gpu_memory_mb"] for r in resource_history]
        gpu_utilization = [r["gpu_utilization"] for r in resource_history]
        
        return {
            "cpu": {
                "mean": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "min": np.min(cpu_usage)
            },
            "memory": {
                "mean": np.mean(memory_usage),
                "max": np.max(memory_usage),
                "min": np.min(memory_usage)
            },
            "gpu_memory": {
                "mean": np.mean(gpu_memory),
                "max": np.max(gpu_memory),
                "min": np.min(gpu_memory)
            },
            "gpu_utilization": {
                "mean": np.mean(gpu_utilization),
                "max": np.max(gpu_utilization),
                "min": np.min(gpu_utilization)
            }
        }
    
    def _analyze_models(self, tasks: List[Dict]) -> Dict:
        """Analyze performance by model"""
        model_performance = {}
        
        for task in tasks:
            if task["status"] != "completed" or not task["metrics"]:
                continue
            
            model = task["model_name"]
            if model not in model_performance:
                model_performance[model] = {
                    "latencies": [],
                    "throughputs": [],
                    "memory_usage": [],
                    "accuracy_scores": []
                }
            
            metrics = task["metrics"]
            if metrics.get("latency_mean", 0) > 0:
                model_performance[model]["latencies"].append(metrics["latency_mean"])
            if metrics.get("throughput_tokens_per_sec", 0) > 0:
                model_performance[model]["throughputs"].append(metrics["throughput_tokens_per_sec"])
            if metrics.get("memory_peak_mb", 0) > 0:
                model_performance[model]["memory_usage"].append(metrics["memory_peak_mb"])
            if metrics.get("accuracy_score", 0) > 0:
                model_performance[model]["accuracy_scores"].append(metrics["accuracy_score"])
        
        # Calculate averages
        model_analysis = {}
        for model, perf in model_performance.items():
            model_analysis[model] = {}
            
            if perf["latencies"]:
                model_analysis[model]["avg_latency"] = np.mean(perf["latencies"])
            if perf["throughputs"]:
                model_analysis[model]["avg_throughput"] = np.mean(perf["throughputs"])
            if perf["memory_usage"]:
                model_analysis[model]["avg_memory"] = np.mean(perf["memory_usage"])
            if perf["accuracy_scores"]:
                model_analysis[model]["avg_accuracy"] = np.mean(perf["accuracy_scores"])
        
        return model_analysis
    
    def _analyze_precisions(self, tasks: List[Dict]) -> Dict:
        """Analyze performance by precision"""
        precision_performance = {}
        
        for task in tasks:
            if task["status"] != "completed" or not task["metrics"]:
                continue
            
            precision = task["precision"]
            if precision not in precision_performance:
                precision_performance[precision] = {
                    "latencies": [],
                    "memory_usage": [],
                    "accuracy_scores": []
                }
            
            metrics = task["metrics"]
            if metrics.get("latency_mean", 0) > 0:
                precision_performance[precision]["latencies"].append(metrics["latency_mean"])
            if metrics.get("memory_peak_mb", 0) > 0:
                precision_performance[precision]["memory_usage"].append(metrics["memory_peak_mb"])
            if metrics.get("accuracy_score", 0) > 0:
                precision_performance[precision]["accuracy_scores"].append(metrics["accuracy_score"])
        
        # Calculate averages and comparisons
        precision_analysis = {}
        for precision, perf in precision_performance.items():
            precision_analysis[precision] = {}
            
            if perf["latencies"]:
                precision_analysis[precision]["avg_latency"] = np.mean(perf["latencies"])
            if perf["memory_usage"]:
                precision_analysis[precision]["avg_memory"] = np.mean(perf["memory_usage"])
            if perf["accuracy_scores"]:
                precision_analysis[precision]["avg_accuracy"] = np.mean(perf["accuracy_scores"])
        
        return precision_analysis
    
    def _analyze_benchmark_types(self, tasks: List[Dict]) -> Dict:
        """Analyze benchmark types"""
        type_stats = {}
        
        for task in tasks:
            benchmark_type = task["benchmark_type"]
            if benchmark_type not in type_stats:
                type_stats[benchmark_type] = {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "avg_execution_time": 0
                }
            
            type_stats[benchmark_type]["total"] += 1
            
            if task["status"] == "completed":
                type_stats[benchmark_type]["completed"] += 1
            elif task["status"] == "failed":
                type_stats[benchmark_type]["failed"] += 1
        
        return type_stats
    
    def _analyze_failures(self, tasks: List[Dict]) -> Dict:
        """Analyze failure patterns"""
        failed_tasks = [t for t in tasks if t["status"] == "failed"]
        
        failure_reasons = {}
        failure_by_model = {}
        failure_by_precision = {}
        
        for task in failed_tasks:
            # Analyze failure reasons
            error_msg = task.get("error_message", "Unknown error")
            if error_msg not in failure_reasons:
                failure_reasons[error_msg] = 0
            failure_reasons[error_msg] += 1
            
            # Analyze failures by model
            model = task["model_name"]
            if model not in failure_by_model:
                failure_by_model[model] = 0
            failure_by_model[model] += 1
            
            # Analyze failures by precision
            precision = task["precision"]
            if precision not in failure_by_precision:
                failure_by_precision[precision] = 0
            failure_by_precision[precision] += 1
        
        return {
            "total_failures": len(failed_tasks),
            "failure_reasons": failure_reasons,
            "failure_by_model": failure_by_model,
            "failure_by_precision": failure_by_precision
        }
    
    def _generate_recommendations(self, tasks: List[Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        completed_tasks = [t for t in tasks if t["status"] == "completed"]
        failed_tasks = [t for t in tasks if t["status"] == "failed"]
        
        # Success rate recommendations
        success_rate = len(completed_tasks) / len(tasks) if tasks else 0
        if success_rate < 0.9:
            recommendations.append(
                f"Success rate is {success_rate:.1%}. Consider investigating failure patterns "
                "and improving error handling."
            )
        
        # Performance recommendations
        if completed_tasks:
            latencies = [t["metrics"]["latency_mean"] for t in completed_tasks 
                        if t["metrics"] and t["metrics"].get("latency_mean", 0) > 0]
            
            if latencies:
                avg_latency = np.mean(latencies)
                if avg_latency > 1.0:
                    recommendations.append(
                        f"Average latency is {avg_latency:.2f}s. Consider using smaller models "
                        "or more aggressive quantization for better performance."
                    )
        
        # Resource utilization recommendations
        resource_history = self.resource_monitor.get_resource_history(minutes=30)
        if resource_history:
            avg_gpu_util = np.mean([r["gpu_utilization"] for r in resource_history])
            if avg_gpu_util < 50:
                recommendations.append(
                    f"GPU utilization is low ({avg_gpu_util:.1f}%). Consider increasing "
                    "batch sizes or running more concurrent tasks."
                )
        
        # Precision recommendations
        precision_analysis = self._analyze_precisions(tasks)
        if "int4" in precision_analysis and "fp16" in precision_analysis:
            int4_memory = precision_analysis["int4"].get("avg_memory", 0)
            fp16_memory = precision_analysis["fp16"].get("avg_memory", 0)
            
            if int4_memory > 0 and fp16_memory > 0:
                memory_savings = (fp16_memory - int4_memory) / fp16_memory
                if memory_savings > 0.5:
                    recommendations.append(
                        f"INT4 quantization provides {memory_savings:.1%} memory savings. "
                        "Consider using INT4 for memory-constrained deployments."
                    )
        
        return recommendations


async def main():
    """Main orchestrator demo"""
    orchestrator = BenchmarkOrchestrator(max_workers=6)
    
    # Test models and configurations
    test_models = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "gpt2",
        "distilgpt2"
    ]
    
    test_precisions = ["fp16", "int8", "int4"]
    test_benchmark_types = ["latency", "memory", "energy", "accuracy"]
    
    # Configuration for different benchmark types
    benchmark_configs = {
        "latency": {"num_runs": 50, "max_tokens": 32},
        "memory": {"test_samples": 10},
        "energy": {"duration_seconds": 30},
        "accuracy": {"num_samples": 500}
    }
    
    try:
        # Start orchestration
        orchestration_task = asyncio.create_task(orchestrator.start_orchestration())
        
        # Submit benchmark suite
        task_ids = orchestrator.submit_benchmark_suite(
            models=test_models,
            precisions=test_precisions,
            benchmark_types=test_benchmark_types,
            config=benchmark_configs
        )
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Monitor progress
        while True:
            status = orchestrator.get_orchestrator_status()
            
            print(f"\nOrchestrator Status:")
            print(f"  Running: {status['is_running']}")
            print(f"  Uptime: {status['uptime_seconds']:.1f}s")
            print(f"  Tasks processed: {status['total_tasks_processed']}")
            print(f"  Scheduler: {status['scheduler']}")
            
            # Check if all tasks are done
            scheduler_stats = status['scheduler']
            if (scheduler_stats['pending'] == 0 and 
                scheduler_stats['queued'] == 0 and
                scheduler_stats['completed'] + scheduler_stats['failed'] >= len(task_ids)):
                print("All tasks completed!")
                break
            
            await asyncio.sleep(5)
        
        # Generate final report
        report = await orchestrator.generate_comprehensive_report()
        
        print(f"\nFinal Report Summary:")
        print(f"  Total tasks: {report['summary']['total_tasks']}")
        print(f"  Success rate: {report['summary']['success_rate']:.1%}")
        print(f"  Total execution time: {report['summary']['total_execution_time']:.1f}s")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
    finally:
        await orchestrator.stop_orchestration()


if __name__ == "__main__":
    asyncio.run(main())