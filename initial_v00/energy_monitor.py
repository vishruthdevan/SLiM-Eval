import subprocess
import time
import json
import os
import signal
import re
from datetime import datetime
import threading

class EnergyMonitor:
    """
    Monitor energy consumption on M3 Max using powermetrics.
    Simplified version that captures system-level power.
    """
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self.monitoring = False
        self.samples = []
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start monitoring energy consumption."""
        self.monitoring = True
        self.samples = []
        self.start_time = time.time()
        
        # Simpler powermetrics command
        cmd = [
            'sudo', 'powermetrics',
            '--samplers', 'tasks',
            '-n', '1',  # Just get one sample to test
            '--show-process-energy'
        ]
        
        try:
            # Test if powermetrics works
            test = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            
            if test.returncode != 0:
                print(f"⚠ Powermetrics test failed: {test.stderr}")
                return False
            
            # Start continuous monitoring
            cmd = [
                'sudo', 'powermetrics',
                '--samplers', 'tasks',
                '-i', str(self.interval_ms),
                '--show-process-energy'
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start reading thread
            self.read_thread = threading.Thread(target=self._read_output, daemon=True)
            self.read_thread.start()
            
            print(f"✓ Energy monitoring started (interval: {self.interval_ms}ms)")
            time.sleep(1)  # Give it time to start
            return True
            
        except Exception as e:
            print(f"✗ Failed to start energy monitoring: {e}")
            return False
    
    def _read_output(self):
        """Read powermetrics output and extract power values."""
        current_sample = {'timestamp': time.time(), 'power_mw': 0}
        
        while self.monitoring and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                # Look for lines with power information
                # Format: "CPU Power: 1234 mW" or similar
                if 'Power:' in line or 'power:' in line.lower():
                    # Extract any number followed by mW
                    matches = re.findall(r'(\d+\.?\d*)\s*mW', line, re.IGNORECASE)
                    if matches:
                        power_value = float(matches[0])
                        if power_value > 0:
                            current_sample['power_mw'] = power_value
                            current_sample['timestamp'] = time.time()
                            self.samples.append(current_sample.copy())
                
                # Also look for total/combined power
                if 'Combined Power' in line or 'Total Power' in line:
                    matches = re.findall(r'(\d+\.?\d*)\s*mW', line, re.IGNORECASE)
                    if matches:
                        power_value = float(matches[0])
                        if power_value > 0:
                            self.samples.append({
                                'timestamp': time.time(),
                                'power_mw': power_value
                            })
                    
            except Exception as e:
                if self.monitoring:
                    print(f"  Warning: Error reading power data: {e}")
                break
    
    def stop(self):
        """Stop monitoring and return results."""
        self.monitoring = False
        self.end_time = time.time()
        
        # Give thread time to finish
        time.sleep(0.5)
        
        if self.process:
            try:
                os.kill(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2)
            except:
                try:
                    os.kill(self.process.pid, signal.SIGKILL)
                except:
                    pass
        
        return self.get_results()
    
    def get_results(self):
        """Calculate energy consumption statistics."""
        if not self.samples or len(self.samples) < 2:
            return {
                'total_energy_joules': 0,
                'avg_power_watts': 0,
                'duration_seconds': self.end_time - self.start_time if self.end_time else 0,
                'num_samples': len(self.samples),
                'error': f'Insufficient samples collected ({len(self.samples)})'
            }
        
        duration = self.end_time - self.start_time
        
        # Calculate energy using trapezoidal integration
        total_energy_mj = 0
        for i in range(1, len(self.samples)):
            time_delta = self.samples[i]['timestamp'] - self.samples[i-1]['timestamp']
            avg_power = (self.samples[i]['power_mw'] + self.samples[i-1]['power_mw']) / 2
            total_energy_mj += avg_power * time_delta
        
        total_energy_joules = total_energy_mj / 1000.0
        avg_power_watts = (total_energy_joules / duration) if duration > 0 else 0
        avg_power_mw = sum(s['power_mw'] for s in self.samples) / len(self.samples)
        
        return {
            'total_energy_joules': total_energy_joules,
            'avg_power_watts': avg_power_watts,
            'avg_power_mw': avg_power_mw,
            'duration_seconds': duration,
            'num_samples': len(self.samples),
        }


class FallbackEnergyMonitor:
    """Fallback monitor using just timing."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.time()
        print("⚠ Using fallback timer (powermetrics unavailable)")
        return True
    
    def stop(self):
        self.end_time = time.time()
        return self.get_results()
    
    def get_results(self):
        duration = self.end_time - self.start_time if self.end_time else 0
        return {
            'total_energy_joules': 0,
            'avg_power_watts': 0,
            'duration_seconds': duration,
            'num_samples': 0,
            'error': 'Powermetrics not available'
        }