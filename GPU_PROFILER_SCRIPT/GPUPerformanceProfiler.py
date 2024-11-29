import GPUtil
import psutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import platform
import cpuinfo
import subprocess
from typing import Dict, Optional

class RTX4060Metrics:
    def __init__(self):
        self.nvidia_smi_path = 'nvidia-smi'
        # RTX 4060 Specifications
        self.cuda_cores = 3072
        self.base_clock = 1830  # MHz
        self.memory_clock = 2250  # MHz effective
        self.memory_bus = 128  # bits
        self.max_power = 115  # Watts
        
    def get_gpu_metrics(self) -> Optional[Dict]:
        try:
            result = subprocess.check_output([
                self.nvidia_smi_path, 
                '--query-gpu=gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics,clocks.current.memory', 
                '--format=csv,noheader,nounits'
            ]).decode()
            
            values = result.strip().split(',')
            metrics = {
                'name': values[0].strip(),
                'temperature': float(values[1]) if values[1].strip() != 'N/A' else 0,
                'gpu_util': float(values[2]) if values[2].strip() != 'N/A' else 0,
                'memory_util': float(values[3]) if values[3].strip() != 'N/A' else 0,
                'memory_used': float(values[4]) if values[4].strip() != 'N/A' else 0,
                'memory_total': float(values[5]) if values[5].strip() != 'N/A' else 0,
                'power_draw': float(values[6]) if values[6].strip() != 'N/A' else 0,
                'gpu_clock': float(values[7]) if values[7].strip() != 'N/A' else 0,
                'memory_clock': float(values[8]) if values[8].strip() != 'N/A' else 0
            }
            
            # Calculate theoretical performance metrics
            metrics['theoretical_tflops'] = (self.cuda_cores * metrics['gpu_clock'] * 2) / 1e6
            metrics['achieved_tflops'] = metrics['theoretical_tflops'] * (metrics['gpu_util'] / 100)
            metrics['memory_bandwidth'] = (metrics['memory_clock'] * self.memory_bus * 2) / 8  # GB/s
            metrics['memory_bandwidth_utilization'] = metrics['memory_bandwidth'] * (metrics['memory_util'] / 100)
            
            return metrics
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return None

class GPUMonitor:
    def __init__(self, log_interval=1):
        self.log_interval = log_interval
        self.data = []
        self.gpu_metrics = RTX4060Metrics()
        
    def get_system_info(self):
        try:
            cpu_info = cpuinfo.get_cpu_info()
            gpu_info = self.gpu_metrics.get_gpu_metrics()
            
            info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'cpu_name': cpu_info['brand_raw'],
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'total_ram': round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                'gpu_name': 'NVIDIA RTX 4060',
                'gpu_architecture': 'Ada Lovelace',
                'compute_capability': '8.9',
                'cuda_cores': self.gpu_metrics.cuda_cores,
                'base_clock': f"{self.gpu_metrics.base_clock} MHz",
                'memory_bus': f"{self.gpu_metrics.memory_bus}-bit",
                'max_power': f"{self.gpu_metrics.max_power}W"
            }
            return info
        except Exception as e:
            print(f"Error getting system info: {e}")
            return None

    def get_metrics(self):
        try:
            gpus = GPUtil.getGPUs()
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            memory = psutil.virtual_memory()
            gpu_metrics = self.gpu_metrics.get_gpu_metrics()
            
            if gpus and gpu_metrics:
                gpu = gpus[0]  # Assuming single GPU system
                metrics = {
                    'timestamp': datetime.now(),
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'gpu_load': gpu_metrics['gpu_util'],
                    'memory_used': gpu_metrics['memory_used'],
                    'memory_total': gpu_metrics['memory_total'],
                    'memory_percent': (gpu_metrics['memory_used'] / gpu_metrics['memory_total'] * 100),
                    'temperature': gpu_metrics['temperature'],
                    'uuid': gpu.uuid,
                    'cpu_avg': sum(cpu_percent) / len(cpu_percent),
                    'cpu_per_core': cpu_percent,
                    'ram_used_percent': memory.percent,
                    'ram_available': round(memory.available / (1024**3), 2),  # GB
                    'gpu_clock': gpu_metrics['gpu_clock'],
                    'memory_clock': gpu_metrics['memory_clock'],
                    'power_draw': gpu_metrics['power_draw'],
                    'theoretical_tflops': gpu_metrics['theoretical_tflops'],
                    'achieved_tflops': gpu_metrics['achieved_tflops'],
                    'memory_bandwidth': gpu_metrics['memory_bandwidth'],
                    'memory_bandwidth_utilization': gpu_metrics['memory_bandwidth_utilization']
                }
                return metrics
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return None

    def start_logging(self, duration_seconds):
        print(f"Starting system monitoring for {duration_seconds} seconds...")
        end_time = time.time() + duration_seconds
        
        # Log system info once at start
        system_info = self.get_system_info()
        
        while time.time() < end_time:
            metrics = self.get_metrics()
            if metrics:
                self.data.append(metrics)
            time.sleep(self.log_interval)
        
        return system_info

    def visualize_data(self, save_path=None):
        if not self.data:
            print("No data to visualize")
            return

        df = pd.DataFrame(self.data)
        
        fig = plt.figure(figsize=(15, 15))
        
        # 1. GPU Usage and Clock Speed
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df['timestamp'], df['gpu_load'], 'b-', label='GPU Usage %')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Usage %', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(df['timestamp'], df['gpu_clock'], 'r-', label='GPU Clock')
        ax2.set_ylabel('Clock Speed (MHz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title('GPU Usage and Clock Speed')
        
        # 2. Memory Usage and Bandwidth
        plt.subplot(3, 2, 2)
        plt.plot(df['timestamp'], df['memory_percent'], 'b-', label='Memory Usage %')
        plt.plot(df['timestamp'], df['memory_bandwidth_utilization'], 'r-', label='Bandwidth Utilization')
        plt.title('Memory Usage and Bandwidth')
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.grid(True)
        plt.legend()

        # 3. Temperature and Power
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(df['timestamp'], df['temperature'], 'g-', label='Temperature')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Temperature (°C)', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        ax3.grid(True)
        
        ax4 = ax3.twinx()
        ax4.plot(df['timestamp'], df['power_draw'], 'r-', label='Power Draw')
        ax4.set_ylabel('Power (W)', color='r')
        ax4.tick_params(axis='y', labelcolor='r')
        plt.title('Temperature and Power Draw')

        # 4. TFLOPS Performance
        plt.subplot(3, 2, 4)
        plt.plot(df['timestamp'], df['theoretical_tflops'], 'b-', label='Theoretical TFLOPS')
        plt.plot(df['timestamp'], df['achieved_tflops'], 'r-', label='Achieved TFLOPS')
        plt.title('Compute Performance')
        plt.xlabel('Time')
        plt.ylabel('TFLOPS')
        plt.grid(True)
        plt.legend()

        # 5. CPU vs GPU Usage
        plt.subplot(3, 2, 5)
        plt.plot(df['timestamp'], df['gpu_load'], 'b-', label='GPU Usage')
        plt.plot(df['timestamp'], df['cpu_avg'], 'r-', label='CPU Usage')
        plt.title('CPU vs GPU Usage')
        plt.xlabel('Time')
        plt.ylabel('Usage %')
        plt.grid(True)
        plt.legend()

        # 6. Memory Clock and Bandwidth
        plt.subplot(3, 2, 6)
        plt.plot(df['timestamp'], df['memory_clock'], 'b-', label='Memory Clock')
        plt.plot(df['timestamp'], df['memory_bandwidth'], 'r-', label='Memory Bandwidth')
        plt.title('Memory Performance')
        plt.xlabel('Time')
        plt.ylabel('MHz / GB/s')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

    def generate_report(self):
        if not self.data:
            print("No data for report generation")
            return

        df = pd.DataFrame(self.data)
        
        report = {
            "GPU Core Metrics": {
                "Average GPU Usage": f"{df['gpu_load'].mean():.2f}%",
                "Peak GPU Usage": f"{df['gpu_load'].max():.2f}%",
                "Average Clock Speed": f"{df['gpu_clock'].mean():.2f} MHz",
                "Peak Clock Speed": f"{df['gpu_clock'].max():.2f} MHz",
                "Average Temperature": f"{df['temperature'].mean():.2f}°C",
                "Peak Temperature": f"{df['temperature'].max():.2f}°C"
            },
            "Compute Performance": {
                "Average Theoretical TFLOPS": f"{df['theoretical_tflops'].mean():.2f}",
                "Peak Theoretical TFLOPS": f"{df['theoretical_tflops'].max():.2f}",
                "Average Achieved TFLOPS": f"{df['achieved_tflops'].mean():.2f}",
                "Peak Achieved TFLOPS": f"{df['achieved_tflops'].max():.2f}",
                "Average Compute Efficiency": f"{(df['achieved_tflops'].mean() / df['theoretical_tflops'].mean() * 100):.2f}%"
            },
            "Memory Metrics": {
                "Average Memory Usage": f"{df['memory_percent'].mean():.2f}%",
                "Peak Memory Usage": f"{df['memory_percent'].max():.2f}%",
                "Average Memory Clock": f"{df['memory_clock'].mean():.2f} MHz",
                "Average Memory Bandwidth": f"{df['memory_bandwidth'].mean():.2f} GB/s",
                "Peak Memory Bandwidth": f"{df['memory_bandwidth'].max():.2f} GB/s"
            },
            "Power Metrics": {
                "Average Power Draw": f"{df['power_draw'].mean():.2f}W",
                "Peak Power Draw": f"{df['power_draw'].max():.2f}W",
                "Power Efficiency": f"{(df['achieved_tflops'].mean() / df['power_draw'].mean()):.2f} TFLOPS/W"
            },
            "System Metrics": {
                "Average CPU Usage": f"{df['cpu_avg'].mean():.2f}%",
                "Peak CPU Usage": f"{df['cpu_avg'].max():.2f}%",
                "Available RAM": f"{df['ram_available'].mean():.2f} GB"
            }
        }
        
        return report

    def save_to_csv(self, filename):
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")

if __name__ == "__main__":
    monitor = GPUMonitor(log_interval=1)
    
    try:
        # Get system info and start monitoring
        system_info = monitor.start_logging(duration_seconds=60)
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        monitor.save_to_csv(f"gpu_metrics_{timestamp}.csv")
        
        # Generate visualizations
        monitor.visualize_data(f"gpu_metrics_{timestamp}.png")
        
        # Generate and print report
        report = monitor.generate_report()
        print("\nPerformance Report:")
        for category, metrics in report.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
                
        if system_info:
            print("\nSystem Information:")
            for key, value in system_info.items():
                print(f"{key}: {value}")
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")