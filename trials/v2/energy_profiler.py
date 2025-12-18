import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


class EnergyProfiler:
    """Base class for energy profiling experiments"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_available = self.check_availability()
    
    def check_availability(self) -> bool:
        """Check if this profiler is available on the system"""
        raise NotImplementedError
    
    def start_monitoring(self):
        """Start energy monitoring"""
        raise NotImplementedError
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return results"""
        raise NotImplementedError


class NVMLProfiler(EnergyProfiler):
    """NVIDIA GPU energy profiling using pynvml"""
    
    def __init__(self):
        super().__init__(
            "NVML",
            "NVIDIA GPU power monitoring using pynvml"
        )
        self.power_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def check_availability(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except:
            return False
    
    def start_monitoring(self):
        if not self.is_available:
            return
        
        import pynvml
        
        self.power_samples = []
        self.monitoring = True
        
        def monitor_power():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self.monitoring:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    self.power_samples.append({
                        'timestamp': time.time(),
                        'power_watts': power_w
                    })
                except:
                    pass
                time.sleep(0.1)  # Sample every 100ms
            
            pynvml.nvmlShutdown()
        
        self.monitor_thread = threading.Thread(target=monitor_power, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        if not self.is_available or not self.monitoring:
            return {"error": "Monitoring not active"}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.power_samples:
            return {"error": "No power samples collected"}
        
        powers = [sample['power_watts'] for sample in self.power_samples]
        timestamps = [sample['timestamp'] for sample in self.power_samples]
        
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        avg_power = np.mean(powers)
        energy_joules = avg_power * duration
        
        return {
            "avg_power_watts": avg_power,
            "min_power_watts": np.min(powers),
            "max_power_watts": np.max(powers),
            "std_power_watts": np.std(powers),
            "duration_seconds": duration,
            "energy_joules": energy_joules,
            "energy_kwh": energy_joules / 3600000,
            "num_samples": len(powers)
        }


class CodeCarbonProfiler(EnergyProfiler):
    """Energy profiling using CodeCarbon"""
    
    def __init__(self):
        super().__init__(
            "CodeCarbon",
            "System-wide energy monitoring using CodeCarbon"
        )
        self.tracker = None
    
    def check_availability(self) -> bool:
        try:
            from codecarbon import EmissionsTracker
            return True
        except ImportError:
            return False
    
    def start_monitoring(self):
        if not self.is_available:
            return
        
        from codecarbon import EmissionsTracker
        
        self.tracker = EmissionsTracker(
            project_name="energy_profiler",
            log_level="warning",
            save_to_file=False
        )
        self.tracker.start()
    
    def stop_monitoring(self) -> Dict:
        if not self.is_available or not self.tracker:
            return {"error": "Tracker not initialized"}
        
        emissions = self.tracker.stop()
        
        if emissions is None:
            return {"error": "No emissions data collected"}
        
        return {
            "emissions_kg_co2": emissions,
            "energy_kwh": emissions * 0.5,  # Rough conversion (varies by region)
            "energy_joules": emissions * 0.5 * 3600000
        }


class PowermetricsMacProfiler(EnergyProfiler):
    """macOS power monitoring using powermetrics"""
    
    def __init__(self):
        super().__init__(
            "Powermetrics",
            "macOS system power monitoring using powermetrics"
        )
        self.process = None
        self.output_file = None
    
    def check_availability(self) -> bool:
        try:
            # Check if we're on macOS and powermetrics is available
            result = subprocess.run(['which', 'powermetrics'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def start_monitoring(self):
        if not self.is_available:
            return
        
        self.output_file = f"/tmp/powermetrics_{int(time.time())}.txt"
        
        # Start powermetrics in background
        self.process = subprocess.Popen([
            'sudo', 'powermetrics',
            '--samplers', 'cpu_power,gpu_power',
            '--sample-rate', '1000',  # 1 second intervals
            '--output-file', self.output_file
        ])
    
    def stop_monitoring(self) -> Dict:
        if not self.is_available or not self.process:
            return {"error": "Process not started"}
        
        # Stop powermetrics
        self.process.terminate()
        self.process.wait()
        
        # Parse output file
        try:
            with open(self.output_file, 'r') as f:
                content = f.read()
            
            # Simple parsing (would need more sophisticated parsing in practice)
            lines = content.split('\n')
            cpu_powers = []
            gpu_powers = []
            
            for line in lines:
                if 'CPU Power' in line:
                    # Extract power value (simplified)
                    try:
                        power = float(line.split(':')[1].strip().split()[0])
                        cpu_powers.append(power)
                    except:
                        pass
                elif 'GPU Power' in line:
                    try:
                        power = float(line.split(':')[1].strip().split()[0])
                        gpu_powers.append(power)
                    except:
                        pass
            
            # Cleanup
            os.remove(self.output_file)
            
            if not cpu_powers and not gpu_powers:
                return {"error": "No power data parsed"}
            
            total_powers = [c + g for c, g in zip(cpu_powers, gpu_powers)]
            if not total_powers:
                total_powers = cpu_powers or gpu_powers
            
            avg_power = np.mean(total_powers)
            duration = len(total_powers)  # Assuming 1 second intervals
            
            return {
                "avg_cpu_power_watts": np.mean(cpu_powers) if cpu_powers else 0,
                "avg_gpu_power_watts": np.mean(gpu_powers) if gpu_powers else 0,
                "avg_total_power_watts": avg_power,
                "duration_seconds": duration,
                "energy_joules": avg_power * duration,
                "num_samples": len(total_powers)
            }
            
        except Exception as e:
            return {"error": f"Failed to parse powermetrics output: {e}"}


class EnergyBenchmark:
    """Energy benchmarking with multiple profiling methods"""
    
    def __init__(self, output_dir: str = "energy_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.profilers = [
            NVMLProfiler(),
            CodeCarbonProfiler(),
            PowermetricsMacProfiler()
        ]
        
        # Filter to available profilers
        self.available_profilers = [p for p in self.profilers if p.is_available]
        
        print(f"Energy Benchmark initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Available profilers: {[p.name for p in self.available_profilers]}")
    
    def run_workload(self, workload_func, workload_name: str, *args, **kwargs):
        """Run a workload while monitoring energy with all available profilers"""
        print(f"\nRunning workload: {workload_name}")
        
        results = {
            "workload": workload_name,
            "timestamp": datetime.now().isoformat(),
            "profilers": {}
        }
        
        # Start all profilers
        for profiler in self.available_profilers:
            try:
                profiler.start_monitoring()
                print(f"  Started {profiler.name} monitoring")
            except Exception as e:
                print(f"  Failed to start {profiler.name}: {e}")
        
        # Run the workload
        start_time = time.time()
        try:
            workload_result = workload_func(*args, **kwargs)
            results["workload_result"] = workload_result
        except Exception as e:
            results["workload_error"] = str(e)
            print(f"  Workload failed: {e}")
        
        end_time = time.time()
        results["workload_duration_s"] = end_time - start_time
        
        # Stop all profilers and collect results
        for profiler in self.available_profilers:
            try:
                profiler_results = profiler.stop_monitoring()
                results["profilers"][profiler.name] = profiler_results
                print(f"  Stopped {profiler.name} monitoring")
            except Exception as e:
                results["profilers"][profiler.name] = {"error": str(e)}
                print(f"  Failed to stop {profiler.name}: {e}")
        
        return results
    
    def dummy_cpu_workload(self, duration: float = 5.0):
        """CPU-intensive dummy workload"""
        print(f"    Running CPU workload for {duration}s...")
        end_time = time.time() + duration
        operations = 0
        
        while time.time() < end_time:
            # Simple CPU-intensive operations
            for i in range(1000):
                _ = i ** 2 + i ** 0.5
            operations += 1000
        
        return {"operations_completed": operations}
    
    def dummy_gpu_workload(self, duration: float = 5.0):
        """GPU-intensive dummy workload (if CUDA available)"""
        print(f"    Running GPU workload for {duration}s...")
        
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
            
            device = torch.device("cuda")
            operations = 0
            end_time = time.time() + duration
            
            while time.time() < end_time:
                # Simple GPU operations
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                operations += 1
            
            return {"gpu_operations_completed": operations}
            
        except ImportError:
            return {"error": "PyTorch not available"}
    
    def mixed_workload(self, duration: float = 5.0):
        """Mixed CPU/GPU workload"""
        print(f"    Running mixed workload for {duration}s...")
        
        cpu_result = self.dummy_cpu_workload(duration / 2)
        gpu_result = self.dummy_gpu_workload(duration / 2)
        
        return {
            "cpu_result": cpu_result,
            "gpu_result": gpu_result
        }
    
    def run_energy_comparison(self):
        """Run energy profiling comparison with different workloads"""
        print(f"\n{'='*60}")
        print("ENERGY PROFILING COMPARISON")
        print(f"{'='*60}")
        
        workloads = [
            (self.dummy_cpu_workload, "CPU_Intensive", 10.0),
            (self.dummy_gpu_workload, "GPU_Intensive", 10.0),
            (self.mixed_workload, "Mixed_Workload", 10.0),
        ]
        
        all_results = []
        
        for workload_func, workload_name, duration in workloads:
            try:
                results = self.run_workload(workload_func, workload_name, duration)
                all_results.append(results)
                
                # Print summary
                print(f"\n  Results for {workload_name}:")
                for profiler_name, profiler_results in results["profilers"].items():
                    if "error" in profiler_results:
                        print(f"    {profiler_name}: Error - {profiler_results['error']}")
                    else:
                        energy = profiler_results.get("energy_joules", "N/A")
                        power = profiler_results.get("avg_power_watts", 
                                                   profiler_results.get("avg_total_power_watts", "N/A"))
                        print(f"    {profiler_name}: Energy={energy}J, Power={power}W")
                
            except Exception as e:
                print(f"Failed to run {workload_name}: {e}")
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = "energy_comparison.json"):
        """Save energy comparison results"""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    def generate_summary(self, results: List[Dict]):
        """Generate energy profiling summary"""
        print(f"\n{'='*60}")
        print("ENERGY PROFILING SUMMARY")
        print(f"{'='*60}")
        
        # Collect all profiler names
        all_profilers = set()
        for result in results:
            all_profilers.update(result["profilers"].keys())
        
        print(f"\nProfilers tested: {', '.join(all_profilers)}")
        
        # Summary table
        print(f"\n{'Workload':<15} {'Profiler':<12} {'Energy (J)':<12} {'Power (W)':<12} {'Status'}")
        print("-" * 65)
        
        for result in results:
            workload = result["workload"]
            for profiler_name, profiler_results in result["profilers"].items():
                if "error" in profiler_results:
                    status = "Failed"
                    energy = "N/A"
                    power = "N/A"
                else:
                    status = "Success"
                    energy = f"{profiler_results.get('energy_joules', 0):.2f}"
                    power = f"{profiler_results.get('avg_power_watts', profiler_results.get('avg_total_power_watts', 0)):.2f}"
                
                print(f"{workload:<15} {profiler_name:<12} {energy:<12} {power:<12} {status}")


def main():
    """Main experimental runner"""
    benchmark = EnergyBenchmark()
    
    if not benchmark.available_profilers:
        print("No energy profilers available on this system")
        return
    
    # Run energy comparison
    results = benchmark.run_energy_comparison()
    
    # Save and summarize results
    benchmark.save_results(results)
    benchmark.generate_summary(results)
    
    print(f"\n{'='*60}")
    print("Energy Profiling Experiments Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()