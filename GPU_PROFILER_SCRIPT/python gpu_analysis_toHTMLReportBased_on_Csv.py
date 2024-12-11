import pandas as pd
import numpy as np
from datetime import datetime

class GPUAnalyzer:
    def __init__(self):
        self.gpu_data = {}
        
    def load_data(self, file_paths):
        """Load and process GPU data from CSV files."""
        for gpu_name, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)
                self.gpu_data[gpu_name] = self.calculate_metrics(df)
            except FileNotFoundError:
                print(f"Error: Could not find file {file_path}")
            except Exception as e:
                print(f"Error processing {gpu_name}: {str(e)}")

    def calculate_metrics(self, df):
        """Calculate comprehensive metrics from GPU data."""
        metrics = {
            'avg_gpu_load': df['gpu_load'].mean(),
            'peak_gpu_load': df['gpu_load'].max(),
            'load_stability': df['gpu_load'].std(),
            'theoretical_tflops': df['theoretical_tflops'].mean(),
            'achieved_tflops': df['achieved_tflops'].mean(),
            'performance_efficiency': (df['achieved_tflops'].mean() / df['theoretical_tflops'].mean()) * 100,
            'idle_temp': df['temperature'].min(),
            'load_temp': df['temperature'].max(),
            'temp_delta': df['temperature'].max() - df['temperature'].min(),
            'temp_variance': df['temperature'].var(),
            'thermal_throttling': len(df[df['temperature'] >= 80]),
            'avg_power': df['power_draw'].mean(),
            'peak_power': df['power_draw'].max(),
            'power_efficiency': df['achieved_tflops'].mean() / df['power_draw'].mean(),
            'power_stability': df['power_draw'].std(),
            'power_temp_correlation': df['power_draw'].corr(df['temperature']),
            'avg_memory_used': df['memory_used'].mean(),
            'peak_memory_used': df['memory_used'].max(),
            'memory_total': df['memory_total'].iloc[0],
            'memory_utilization': (df['memory_used'].mean() / df['memory_total'].iloc[0]) * 100,
            'memory_bandwidth_util': df['memory_bandwidth_utilization'].mean()
        }
        return metrics

    def create_html_table(self, metric_group):
        """Create HTML table for a group of metrics."""
        headers = ['Metric'] + list(self.gpu_data.keys())
        table_html = ['<table>']
        table_html.append('<tr>')
        for header in headers:
            table_html.append(f'<th>{header}</th>')
        table_html.append('</tr>')
        
        for metric, display_name in metric_group:
            row = ['<tr>']
            row.append(f'<td>{display_name}</td>')
            
            for gpu in self.gpu_data.keys():
                value = self.gpu_data[gpu][metric]
                
                if 'temp' in metric and 'correlation' not in metric:
                    formatted_value = f"{value:.1f}°C"
                elif 'percent' in metric or 'efficiency' in metric:
                    formatted_value = f"{value:.2f}%"
                elif 'power' in metric and 'correlation' not in metric:
                    formatted_value = f"{value:.2f}W"
                elif 'memory' in metric and 'bandwidth' not in metric:
                    formatted_value = f"{value:,.0f} MB"
                elif 'tflops' in metric.lower():
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2f}"
                
                row.append(f'<td>{formatted_value}</td>')
            
            row.append('</tr>')
            table_html.append(''.join(row))
        
        table_html.append('</table>')
        return '\n'.join(table_html)

    def generate_report(self, output_file):
        """Generate complete HTML report."""
        sections = {
            'Performance Metrics': [
                ('avg_gpu_load', 'Average GPU Load'),
                ('peak_gpu_load', 'Peak GPU Load'),
                ('load_stability', 'Load Stability (σ)'),
                ('theoretical_tflops', 'Theoretical TFLOPS'),
                ('achieved_tflops', 'Average Achieved TFLOPS'),
                ('performance_efficiency', 'Performance Efficiency')
            ],
            'Thermal Analysis': [
                ('idle_temp', 'Idle Temperature'),
                ('load_temp', 'Load Temperature'),
                ('temp_delta', 'Temperature Delta'),
                ('temp_variance', 'Temperature Variance'),
                ('thermal_throttling', 'Thermal Throttling Instances')
            ],
            'Power Analysis': [
                ('avg_power', 'Average Power Draw'),
                ('peak_power', 'Peak Power Draw'),
                ('power_efficiency', 'Power Efficiency (TFLOPS/W)'),
                ('power_stability', 'Power Draw Stability (σ)'),
                ('power_temp_correlation', 'Power/Temp Correlation')
            ],
            'Memory Performance': [
                ('avg_memory_used', 'Average Memory Used'),
                ('peak_memory_used', 'Peak Memory Used'),
                ('memory_total', 'Total Memory'),
                ('memory_utilization', 'Memory Utilization'),
                ('memory_bandwidth_util', 'Memory Bandwidth Utilization')
            ]
        }

        html_parts = []
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f7fa; }
        .report-header { text-align: center; padding: 20px; margin-bottom: 30px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; border-radius: 8px; }
        .section { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .section-title { color: #1e3c72; margin-bottom: 15px; border-bottom: 2px solid #1e3c72; padding-bottom: 5px; }
    </style>
</head>
<body>
        """)
        
        html_parts.append(f'''
    <div class="report-header">
        <h1>GPU Performance Analysis Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
        ''')

        for section_title, metrics in sections.items():
            html_parts.append(f'''
    <div class="section">
        <h2 class="section-title">{section_title}</h2>
        {self.create_html_table(metrics)}
    </div>
            ''')

        html_parts.append("""
</body>
</html>
        """)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))
            print(f"Report successfully generated: {output_file}")
        except Exception as e:
            print(f"Error writing report: {str(e)}")

def main():
    # Define file paths for each GPU
    gpu_files = {
        'RTX 4060': 'gpu_metrics_20241103_165218.csv',
        'RTX 3060': 'gpu_metrics_20241103_183532Pranay.csv',
        'RTX 4000 Ada': 'gpu_metrics_20241103_Andrew.csv',
        'RTX 3080': 'gpu_metrics_20241104_054223Patrick.csv'
    }

    analyzer = GPUAnalyzer()
    analyzer.load_data(gpu_files)
    analyzer.generate_report('gpu_performance_report.html')

if __name__ == "__main__":
    main()