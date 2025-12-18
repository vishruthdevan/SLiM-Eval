import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StatisticalPlotter:
    """Advanced statistical plotting capabilities"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def create_distribution_plot(self, data: List[float], title: str, 
                               xlabel: str, bins: int = 30) -> plt.Figure:
        """Create comprehensive distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(f'Distribution Analysis: {title}', fontsize=16, fontweight='bold')
        
        data_array = np.array(data)
        
        # Histogram with KDE
        axes[0, 0].hist(data_array, bins=bins, density=True, alpha=0.7, 
                       color=self.colors[0], edgecolor='black')
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data_array)
        x_range = np.linspace(data_array.min(), data_array.max(), 100)
        axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        axes[0, 0].set_title('Histogram with KDE')
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        box_plot = axes[0, 1].boxplot(data_array, patch_artist=True, 
                                     boxprops=dict(facecolor=self.colors[1]))
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(xlabel)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistical annotations
        q1, median, q3 = np.percentile(data_array, [25, 50, 75])
        axes[0, 1].text(1.1, median, f'Median: {median:.3f}', 
                       transform=axes[0, 1].transData, fontsize=10)
        axes[0, 1].text(1.1, q1, f'Q1: {q1:.3f}', 
                       transform=axes[0, 1].transData, fontsize=10)
        axes[0, 1].text(1.1, q3, f'Q3: {q3:.3f}', 
                       transform=axes[0, 1].transData, fontsize=10)
        
        # Q-Q plot for normality assessment
        from scipy import stats
        stats.probplot(data_array, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Violin plot
        violin_parts = axes[1, 1].violinplot([data_array], positions=[1], 
                                           showmeans=True, showmedians=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(self.colors[2])
            pc.set_alpha(0.7)
        
        axes[1, 1].set_title('Violin Plot')
        axes[1, 1].set_ylabel(xlabel)
        axes[1, 1].set_xticks([1])
        axes[1, 1].set_xticklabels(['Data'])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        skew_val = stats.skew(data_array)
        kurt_val = stats.kurtosis(data_array)
        
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nSkewness: {skew_val:.3f}\nKurtosis: {kurt_val:.3f}'
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                  title: str = "Correlation Matrix") -> plt.Figure:
        """Create advanced correlation heatmap"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                             cmap='RdBu_r', center=0, square=True, 
                             linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Add significance indicators (simplified)
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, 
                                                 edgecolor='gold', lw=3))
        
        plt.tight_layout()
        return fig
    
    def create_regression_plot(self, x_data: List[float], y_data: List[float], 
                              x_label: str, y_label: str, title: str) -> plt.Figure:
        """Create comprehensive regression analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(f'Regression Analysis: {title}', fontsize=16, fontweight='bold')
        
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # Scatter plot with regression line
        axes[0, 0].scatter(x_array, y_array, alpha=0.6, color=self.colors[0])
        
        # Fit regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_array, y_array)
        line_x = np.linspace(x_array.min(), x_array.max(), 100)
        line_y = slope * line_x + intercept
        
        axes[0, 0].plot(line_x, line_y, 'r-', linewidth=2, 
                       label=f'R² = {r_value**2:.3f}')
        
        # Add confidence interval
        residuals = y_array - (slope * x_array + intercept)
        mse = np.mean(residuals**2)
        confidence_interval = 1.96 * np.sqrt(mse)
        
        axes[0, 0].fill_between(line_x, line_y - confidence_interval, 
                               line_y + confidence_interval, alpha=0.2, color='red')
        
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel(y_label)
        axes[0, 0].set_title('Scatter Plot with Regression')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        predicted = slope * x_array + intercept
        residuals = y_array - predicted
        
        axes[0, 1].scatter(predicted, residuals, alpha=0.6, color=self.colors[1])
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 1].hist(residuals, bins=20, density=True, alpha=0.7, 
                       color=self.colors[2], edgecolor='black')
        
        # Add normal curve
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        axes[1, 1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Curve')
        
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add regression statistics
        stats_text = f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR²: {r_value**2:.4f}\nP-value: {p_value:.4f}'
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig


class PerformancePlotter:
    """Specialized plotting for performance metrics"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("Set2", 10)
    
    def create_performance_comparison(self, data: pd.DataFrame, 
                                    group_by: str, metrics: List[str]) -> plt.Figure:
        """Create comprehensive performance comparison plots"""
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Performance Comparison by {group_by}', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if metric in data.columns:
                # Create box plot with individual points
                groups = data.groupby(group_by)[metric].apply(list).to_dict()
                
                # Box plot
                box_data = [group_data for group_data in groups.values() if len(group_data) > 0]
                box_labels = [label for label, group_data in groups.items() if len(group_data) > 0]
                
                if box_data:
                    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                    
                    # Color boxes
                    for patch, color in zip(bp['boxes'], self.colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add individual points
                    for j, (label, group_data) in enumerate(groups.items()):
                        if len(group_data) > 0:
                            y_data = group_data
                            x_data = np.random.normal(j + 1, 0.04, size=len(y_data))
                            ax.scatter(x_data, y_data, alpha=0.6, s=20, color='black')
                    
                    # Add mean markers
                    means = [np.mean(group_data) for group_data in box_data]
                    ax.scatter(range(1, len(means) + 1), means, marker='D', 
                              s=50, color='red', label='Mean', zorder=10)
                    
                    ax.set_title(f'{metric}')
                    ax.set_ylabel(metric)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Rotate x-labels if needed
                    if len(max(box_labels, key=len)) > 10:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Remove empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        return fig
    
    def create_performance_radar(self, data: pd.DataFrame, group_by: str, 
                               metrics: List[str]) -> plt.Figure:
        """Create radar chart for performance comparison"""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi, subplot_kw=dict(projection='polar'))
        
        # Prepare data
        groups = data[group_by].unique()
        n_metrics = len(metrics)
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_data = {}
        for metric in metrics:
            if metric in data.columns:
                metric_data = data[metric].dropna()
                if len(metric_data) > 0:
                    min_val, max_val = metric_data.min(), metric_data.max()
                    if max_val > min_val:
                        normalized_data[metric] = (metric_data - min_val) / (max_val - min_val)
                    else:
                        normalized_data[metric] = pd.Series([0.5] * len(metric_data), index=metric_data.index)
        
        # Plot each group
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group]
            values = []
            
            for metric in metrics:
                if metric in normalized_data:
                    group_metric_data = normalized_data[metric][group_data.index]
                    values.append(group_metric_data.mean() if len(group_metric_data) > 0 else 0)
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=group, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(f'Performance Radar Chart by {group_by}', size=16, fontweight='bold', pad=20)
        
        return fig
    
    def create_efficiency_frontier(self, data: pd.DataFrame, x_metric: str, 
                                 y_metric: str, group_by: str) -> plt.Figure:
        """Create efficiency frontier plot (Pareto front)"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if x_metric not in data.columns or y_metric not in data.columns:
            ax.text(0.5, 0.5, f'Metrics {x_metric} or {y_metric} not found', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Plot data points by group
        groups = data[group_by].unique() if group_by in data.columns else ['All Data']
        
        for i, group in enumerate(groups):
            if group_by in data.columns:
                group_data = data[data[group_by] == group]
            else:
                group_data = data
            
            x_vals = group_data[x_metric].dropna()
            y_vals = group_data[y_metric].dropna()
            
            # Align data
            common_idx = x_vals.index.intersection(y_vals.index)
            x_aligned = x_vals[common_idx]
            y_aligned = y_vals[common_idx]
            
            if len(x_aligned) > 0:
                ax.scatter(x_aligned, y_aligned, alpha=0.7, s=60, 
                          color=self.colors[i % len(self.colors)], label=group)
        
        # Calculate and plot Pareto frontier
        all_x = data[x_metric].dropna()
        all_y = data[y_metric].dropna()
        common_idx = all_x.index.intersection(all_y.index)
        
        if len(common_idx) > 2:
            x_pareto = all_x[common_idx].values
            y_pareto = all_y[common_idx].values
            
            # Find Pareto frontier (assuming higher is better for both metrics)
            pareto_indices = []
            for i in range(len(x_pareto)):
                is_pareto = True
                for j in range(len(x_pareto)):
                    if i != j and x_pareto[j] >= x_pareto[i] and y_pareto[j] >= y_pareto[i]:
                        if x_pareto[j] > x_pareto[i] or y_pareto[j] > y_pareto[i]:
                            is_pareto = False
                            break
                if is_pareto:
                    pareto_indices.append(i)
            
            if pareto_indices:
                pareto_x = x_pareto[pareto_indices]
                pareto_y = y_pareto[pareto_indices]
                
                # Sort for plotting
                sorted_indices = np.argsort(pareto_x)
                pareto_x_sorted = pareto_x[sorted_indices]
                pareto_y_sorted = pareto_y[sorted_indices]
                
                ax.plot(pareto_x_sorted, pareto_y_sorted, 'r--', linewidth=2, 
                       label='Pareto Frontier', alpha=0.8)
                ax.scatter(pareto_x_sorted, pareto_y_sorted, color='red', s=100, 
                          marker='*', label='Pareto Optimal', zorder=10)
        
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f'Efficiency Frontier: {y_metric} vs {x_metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class InteractivePlotter:
    """Interactive plotting using Plotly"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_interactive_dashboard(self, data: pd.DataFrame, 
                                   output_file: str = "dashboard.html") -> str:
        """Create comprehensive interactive dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Performance Overview', 'Model Comparison', 
                          'Precision Analysis', 'Trend Analysis',
                          'Correlation Matrix', 'Distribution Analysis'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "histogram"}]]
        )
        
        # 1. Performance Overview (Multi-metric line plot)
        if 'timestamp' in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:3]
            for i, col in enumerate(numeric_cols):
                fig.add_trace(
                    go.Scatter(x=data['timestamp'], y=data[col], 
                             mode='lines+markers', name=col,
                             line=dict(color=self.colors[i % len(self.colors)])),
                    row=1, col=1
                )
        
        # 2. Model Comparison (Bar chart)
        if 'model_name' in data.columns:
            model_performance = data.groupby('model_name').agg({
                col: 'mean' for col in data.select_dtypes(include=[np.number]).columns[:3]
            }).reset_index()
            
            for i, col in enumerate(model_performance.columns[1:]):
                fig.add_trace(
                    go.Bar(x=model_performance['model_name'], y=model_performance[col],
                          name=col, marker_color=self.colors[i % len(self.colors)]),
                    row=1, col=2
                )
        
        # 3. Precision Analysis (Box plot)
        if 'precision' in data.columns:
            numeric_col = data.select_dtypes(include=[np.number]).columns[0]
            for precision in data['precision'].unique():
                precision_data = data[data['precision'] == precision][numeric_col].dropna()
                fig.add_trace(
                    go.Box(y=precision_data, name=precision),
                    row=2, col=1
                )
        
        # 4. Trend Analysis (Scatter plot with trendline)
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:2]
        if len(numeric_cols) >= 2:
            fig.add_trace(
                go.Scatter(x=data[numeric_cols[0]], y=data[numeric_cols[1]],
                          mode='markers', name='Data Points',
                          marker=dict(color=data.index, colorscale='Viridis')),
                row=2, col=2
            )
        
        # 5. Correlation Matrix (Heatmap)
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                          y=corr_matrix.columns, colorscale='RdBu',
                          zmid=0, text=corr_matrix.values, texttemplate="%{text:.2f}"),
                row=3, col=1
            )
        
        # 6. Distribution Analysis (Histogram)
        if not numeric_data.empty:
            first_numeric_col = numeric_data.columns[0]
            fig.add_trace(
                go.Histogram(x=data[first_numeric_col], nbinsx=30, 
                           name='Distribution', marker_color=self.colors[0]),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="SLiM-Eval Interactive Dashboard",
            title_x=0.5,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save to HTML
        pyo.plot(fig, filename=output_file, auto_open=False)
        
        return output_file
    
    def create_3d_performance_plot(self, data: pd.DataFrame, x_col: str, 
                                  y_col: str, z_col: str, color_col: str = None) -> go.Figure:
        """Create 3D performance visualization"""
        if not all(col in data.columns for col in [x_col, y_col, z_col]):
            return go.Figure().add_annotation(text="Required columns not found")
        
        # Prepare color data
        if color_col and color_col in data.columns:
            color_data = data[color_col]
            color_title = color_col
        else:
            color_data = data.index
            color_title = "Index"
        
        fig = go.Figure(data=go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=8,
                color=color_data,
                colorscale='Viridis',
                colorbar=dict(title=color_title),
                opacity=0.8
            ),
            text=data.index,
            hovertemplate=f'<b>%{{text}}</b><br>' +
                         f'{x_col}: %{{x}}<br>' +
                         f'{y_col}: %{{y}}<br>' +
                         f'{z_col}: %{{z}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Performance Analysis: {x_col} vs {y_col} vs {z_col}',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template="plotly_white"
        )
        
        return fig


class ReportGenerator:
    """Comprehensive report generation"""
    
    def __init__(self, output_dir: str = "visualization_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.statistical_plotter = StatisticalPlotter()
        self.performance_plotter = PerformancePlotter()
        self.interactive_plotter = InteractivePlotter()
    
    def generate_comprehensive_report(self, data: pd.DataFrame, 
                                    analysis_results: Dict = None) -> Dict[str, str]:
        """Generate comprehensive visualization report"""
        print("Generating comprehensive visualization report...")
        
        report_files = {}
        
        # 1. Statistical Analysis Report
        if analysis_results and 'statistical_analysis' in analysis_results:
            stats_pdf = self._generate_statistical_report(data, analysis_results['statistical_analysis'])
            report_files['statistical_report'] = stats_pdf
        
        # 2. Performance Analysis Report
        performance_pdf = self._generate_performance_report(data)
        report_files['performance_report'] = performance_pdf
        
        # 3. Interactive Dashboard
        dashboard_html = self.interactive_plotter.create_interactive_dashboard(
            data, str(self.output_dir / "interactive_dashboard.html")
        )
        report_files['interactive_dashboard'] = dashboard_html
        
        # 4. Executive Summary
        summary_pdf = self._generate_executive_summary(data, analysis_results)
        report_files['executive_summary'] = summary_pdf
        
        print(f"Report generation completed. Files saved to: {self.output_dir}")
        return report_files
    
    def _generate_statistical_report(self, data: pd.DataFrame, 
                                   statistical_analysis: Dict) -> str:
        """Generate statistical analysis report"""
        pdf_file = self.output_dir / "statistical_analysis_report.pdf"
        
        with PdfPages(pdf_file) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, 'Statistical Analysis Report', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.6, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.5, f'Total Records: {len(data)}', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Distribution plots for key metrics
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns[:6]:  # Limit to first 6 columns
                column_data = data[column].dropna().tolist()
                if len(column_data) > 2:
                    fig = self.statistical_plotter.create_distribution_plot(
                        column_data, column, column
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                correlation_matrix = data[numeric_columns].corr()
                fig = self.statistical_plotter.create_correlation_heatmap(
                    correlation_matrix, "Feature Correlation Matrix"
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Regression plots for key relationships
            if 'correlations' in statistical_analysis:
                correlations = statistical_analysis['correlations']
                
                for pair_name, corr_data in list(correlations.items())[:3]:  # First 3 pairs
                    if 'pearson' in corr_data and corr_data['pearson'].get('significant', False):
                        # Extract column names from pair_name
                        col1, col2 = pair_name.split('_vs_')
                        
                        if col1 in data.columns and col2 in data.columns:
                            # Align data
                            aligned_data = data[[col1, col2]].dropna()
                            if len(aligned_data) > 2:
                                fig = self.statistical_plotter.create_regression_plot(
                                    aligned_data[col1].tolist(),
                                    aligned_data[col2].tolist(),
                                    col1, col2, f"{col1} vs {col2}"
                                )
                                pdf.savefig(fig, bbox_inches='tight')
                                plt.close(fig)
        
        return str(pdf_file)
    
    def _generate_performance_report(self, data: pd.DataFrame) -> str:
        """Generate performance analysis report"""
        pdf_file = self.output_dir / "performance_analysis_report.pdf"
        
        with PdfPages(pdf_file) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, 'Performance Analysis Report', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.6, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Performance metrics
            performance_metrics = [col for col in data.columns 
                                 if any(keyword in col.lower() 
                                       for keyword in ['latency', 'memory', 'energy', 'accuracy', 'throughput'])]
            
            # Model comparison
            if 'model_name' in data.columns and performance_metrics:
                fig = self.performance_plotter.create_performance_comparison(
                    data, 'model_name', performance_metrics[:4]
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Precision comparison
            if 'precision' in data.columns and performance_metrics:
                fig = self.performance_plotter.create_performance_comparison(
                    data, 'precision', performance_metrics[:4]
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Radar chart
            if 'model_name' in data.columns and len(performance_metrics) >= 3:
                fig = self.performance_plotter.create_performance_radar(
                    data, 'model_name', performance_metrics[:6]
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Efficiency frontiers
            if len(performance_metrics) >= 2:
                for i in range(0, len(performance_metrics)-1, 2):
                    if i+1 < len(performance_metrics):
                        group_by = 'precision' if 'precision' in data.columns else 'model_name'
                        if group_by in data.columns:
                            fig = self.performance_plotter.create_efficiency_frontier(
                                data, performance_metrics[i], performance_metrics[i+1], group_by
                            )
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
        
        return str(pdf_file)
    
    def _generate_executive_summary(self, data: pd.DataFrame, 
                                  analysis_results: Dict = None) -> str:
        """Generate executive summary report"""
        pdf_file = self.output_dir / "executive_summary.pdf"
        
        with PdfPages(pdf_file) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, 'Executive Summary', 
                   ha='center', va='center', fontsize=28, fontweight='bold')
            ax.text(0.5, 0.7, 'SLiM-Eval Benchmark Analysis', 
                   ha='center', va='center', fontsize=18)
            ax.text(0.5, 0.6, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                   ha='center', va='center', fontsize=12)
            
            # Key statistics
            stats_text = f"""
            Dataset Overview:
            • Total Records: {len(data):,}
            • Features Analyzed: {len(data.columns)}
            • Date Range: {data['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in data.columns else 'N/A'} to {data['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in data.columns else 'N/A'}
            """
            
            if 'model_name' in data.columns:
                stats_text += f"• Models Evaluated: {data['model_name'].nunique()}\n"
            
            if 'precision' in data.columns:
                stats_text += f"• Precision Types: {', '.join(data['precision'].unique())}\n"
            
            ax.text(0.1, 0.4, stats_text, ha='left', va='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Key findings page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.9, 'Key Findings', 
                   ha='center', va='center', fontsize=20, fontweight='bold')
            
            findings_text = "Key Performance Insights:\n\n"
            
            # Performance insights
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        findings_text += f"• {col}: Mean = {col_data.mean():.3f}, Std = {col_data.std():.3f}\n"
            
            # Model comparison insights
            if 'model_name' in data.columns and len(numeric_cols) > 0:
                findings_text += "\nModel Performance Ranking:\n"
                
                for col in numeric_cols[:2]:
                    model_means = data.groupby('model_name')[col].mean().sort_values(ascending=False)
                    findings_text += f"\n{col}:\n"
                    for i, (model, value) in enumerate(model_means.head(3).items()):
                        findings_text += f"  {i+1}. {model}: {value:.3f}\n"
            
            # Recommendations
            if analysis_results and 'recommendations' in analysis_results:
                findings_text += "\nKey Recommendations:\n"
                for i, rec in enumerate(analysis_results['recommendations'][:5], 1):
                    findings_text += f"{i}. {rec}\n"
            
            ax.text(0.05, 0.8, findings_text, ha='left', va='top', fontsize=10, 
                   wrap=True, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        return str(pdf_file)


class VisualizationEngine:
    """Main visualization engine"""
    
    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        print(f"Visualization Engine initialized")
        print(f"Output directory: {self.output_dir}")
    
    def create_comprehensive_visualizations(self, data: pd.DataFrame, 
                                          analysis_results: Dict = None) -> Dict[str, str]:
        """Create comprehensive visualization suite"""
        print("Creating comprehensive visualizations...")
        
        # Generate all reports
        report_files = self.report_generator.generate_comprehensive_report(data, analysis_results)
        
        # Create additional interactive plots
        interactive_files = self._create_interactive_plots(data)
        report_files.update(interactive_files)
        
        # Create summary index
        index_file = self._create_index_file(report_files)
        report_files['index'] = index_file
        
        print("Visualization suite completed!")
        return report_files
    
    def _create_interactive_plots(self, data: pd.DataFrame) -> Dict[str, str]:
        """Create additional interactive plots"""
        interactive_files = {}
        
        # 3D performance plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            fig_3d = self.report_generator.interactive_plotter.create_3d_performance_plot(
                data, numeric_cols[0], numeric_cols[1], numeric_cols[2],
                'model_name' if 'model_name' in data.columns else None
            )
            
            plot_3d_file = self.output_dir / "3d_performance_plot.html"
            fig_3d.write_html(str(plot_3d_file))
            interactive_files['3d_performance'] = str(plot_3d_file)
        
        return interactive_files
    
    def _create_index_file(self, report_files: Dict[str, str]) -> str:
        """Create HTML index file for all reports"""
        index_file = self.output_dir / "index.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SLiM-Eval Visualization Reports</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .report-link {{ 
                    display: block; 
                    margin: 10px 0; 
                    padding: 10px; 
                    background-color: #ecf0f1; 
                    text-decoration: none; 
                    border-radius: 5px; 
                }}
                .report-link:hover {{ background-color: #bdc3c7; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>SLiM-Eval Visualization Reports</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Available Reports</h2>
        """
        
        report_descriptions = {
            'statistical_report': 'Statistical Analysis Report - Comprehensive statistical analysis with distribution plots and correlation analysis',
            'performance_report': 'Performance Analysis Report - Detailed performance comparisons and efficiency analysis',
            'executive_summary': 'Executive Summary - High-level overview and key findings',
            'interactive_dashboard': 'Interactive Dashboard - Interactive plots and data exploration',
            '3d_performance': '3D Performance Visualization - Three-dimensional performance analysis'
        }
        
        for report_key, file_path in report_files.items():
            if report_key != 'index':
                file_name = Path(file_path).name
                description = report_descriptions.get(report_key, 'Analysis report')
                
                html_content += f"""
                <a href="{file_name}" class="report-link">
                    <strong>{report_key.replace('_', ' ').title()}</strong><br>
                    <small>{description}</small>
                </a>
                """
        
        html_content += """
            <h2>Report Descriptions</h2>
            <ul>
                <li><strong>Statistical Analysis Report:</strong> In-depth statistical analysis including distribution analysis, normality tests, and correlation studies.</li>
                <li><strong>Performance Analysis Report:</strong> Comprehensive performance comparisons across models, precisions, and benchmark types.</li>
                <li><strong>Executive Summary:</strong> High-level overview with key findings and actionable recommendations.</li>
                <li><strong>Interactive Dashboard:</strong> Interactive visualizations for data exploration and analysis.</li>
                <li><strong>3D Performance Visualization:</strong> Three-dimensional analysis of performance relationships.</li>
            </ul>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #bdc3c7;">
                <p><em>Generated by SLiM-Eval Visualization Engine </em></p>
            </footer>
        </body>
        </html>
        """
        
        with open(index_file, 'w') as f:
            f.write(html_content)
        
        return str(index_file)


def main():
    """Main visualization demo"""
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Create visualization engine
    viz_engine = VisualizationEngine()
    
    # Generate comprehensive visualizations
    report_files = viz_engine.create_comprehensive_visualizations(sample_data)
    
    print(f"\nVisualization suite completed!")
    print(f"Reports generated:")
    for report_type, file_path in report_files.items():
        print(f"  {report_type}: {file_path}")
    
    print(f"\nOpen {report_files['index']} in your browser to view all reports.")


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    models = ['model_small', 'model_medium', 'model_large']
    precisions = ['fp16', 'int8', 'int4']
    
    data = []
    
    for i in range(150):
        model = np.random.choice(models)
        precision = np.random.choice(precisions)
        
        # Simulate realistic relationships
        model_factor = {'model_small': 0.5, 'model_medium': 1.0, 'model_large': 2.0}[model]
        precision_factor = {'fp16': 1.0, 'int8': 0.7, 'int4': 0.5}[precision]
        
        record = {
            'model_name': model,
            'precision': precision,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
            'latency_mean': max(0.01, np.random.normal(0.1 * model_factor / precision_factor, 0.02)),
            'memory_peak_mb': max(100, np.random.normal(1000 * model_factor * precision_factor, 200)),
            'energy_joules': max(1, np.random.normal(50 * model_factor / precision_factor, 10)),
            'accuracy_score': max(0.0, min(1.0, np.random.normal(0.8 - (1 - precision_factor) * 0.1, 0.05))),
            'throughput_tokens_per_sec': max(1, np.random.normal(100 / model_factor * precision_factor, 20))
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()