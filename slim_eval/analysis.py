"""Analysis and visualization utilities for SLiM-Eval."""

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes and visualizes benchmark results."""

    def __init__(self, input_dir: Path, output_dir: Path, accuracy_tasks: list):
        """Initialize results analyzer.

        Args:
            input_dir: Directory containing benchmark results to analyze.
            output_dir: Directory for saving analysis outputs.
            accuracy_tasks: List of accuracy task names.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.accuracy_tasks = accuracy_tasks
        # Create plots subdirectory
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_results_from_json(self) -> pd.DataFrame:
        """Load all benchmark results from JSON files in model directories.

        Returns:
            DataFrame with all results combined.
        """
        all_results = []

        # Find all model directories (first level)
        for model_dir in self.input_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Look for model_precision subdirectories (second level)
            for precision_dir in model_dir.iterdir():
                if not precision_dir.is_dir():
                    continue

                # Skip if it doesn't contain JSON files
                if not any(
                    f.suffix == ".json" for f in precision_dir.iterdir() if f.is_file()
                ):
                    continue

                result = {}

                # Load performance.json
                performance_file = precision_dir / "performance.json"
                if performance_file.exists():
                    with open(performance_file, "r") as f:
                        result.update(json.load(f))

                # Load energy.json
                energy_file = precision_dir / "energy.json"
                if energy_file.exists():
                    with open(energy_file, "r") as f:
                        energy_data = json.load(f)
                        # Add energy fields to result
                        for key, value in energy_data.items():
                            if key not in result:  # Avoid overwriting metadata
                                result[key] = value

                # Load accuracy JSON files
                for task in self.accuracy_tasks:
                    accuracy_file = precision_dir / f"{task}.json"
                    if accuracy_file.exists():
                        with open(accuracy_file, "r") as f:
                            accuracy_data = json.load(f)
                            result[f"{task}_accuracy"] = accuracy_data.get("accuracy")

                if result:
                    all_results.append(result)

        if not all_results:
            logger.warning("No results found in output directory")
            return pd.DataFrame()

        return pd.DataFrame(all_results)

    def analyze_results(self):
        """Run complete analysis suite."""
        results_df = self.load_results_from_json()

        if results_df.empty:
            logger.warning("No results to analyze")
            return

        logger.info("#" * 70)
        logger.info("COMPLETE RESULTS SUMMARY")
        logger.info("#" * 70)

        display_columns = [
            "model",
            "num_parameters_b",
            "precision",
            "mean_latency_s",
            "mean_peak_mem_mb",
            "energy_kwh",
            "avg_power_watts",
        ] + [col for col in results_df.columns if "accuracy" in col]

        logger.info("\n" + results_df[display_columns].round(4).to_string(index=False))

        # Run all analyses
        self.analyze_quantization_impact(results_df)
        self.generate_all_visualizations(results_df)
        self.generate_summary_statistics(results_df)
        self.generate_results_table(results_df)
        self.generate_executive_summary(results_df)
        self.export_json(results_df)

    def analyze_quantization_impact(self, results_df: pd.DataFrame):
        """Analyze the impact of quantization compared to FP16 baseline."""
        analysis_results = []
        for model in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model]
            fp16_data = model_data[model_data["precision"] == "fp16"]
            if len(fp16_data) == 0:
                continue

            fp16_latency = fp16_data["mean_latency_s"].values[0]
            fp16_memory = fp16_data["mean_peak_mem_mb"].values[0]
            fp16_energy = (
                fp16_data["energy_joules"].values[0]
                if "energy_joules" in fp16_data.columns
                else 0
            )

            for _, row in model_data.iterrows():
                if row["precision"] != "fp16":
                    analysis_row = {
                        "model": str(model).split("/")[-1],
                        "precision": row["precision"],
                        "speedup": fp16_latency / row["mean_latency_s"],
                        "memory_reduction_pct": (
                            1 - row["mean_peak_mem_mb"] / fp16_memory
                        )
                        * 100,
                    }

                    if "energy_joules" in row and fp16_energy > 0:
                        analysis_row["energy_reduction_pct"] = (
                            1 - row["energy_joules"] / fp16_energy
                        ) * 100

                    for task in self.accuracy_tasks:
                        acc_col = f"{task}_accuracy"
                        if acc_col in fp16_data.columns and acc_col in row:
                            fp16_acc = fp16_data[acc_col].values[0]
                            quant_acc = row[acc_col]
                            if fp16_acc > 0:
                                analysis_row[f"{task}_acc_drop_pct"] = (
                                    (fp16_acc - quant_acc) / fp16_acc
                                ) * 100

                    analysis_results.append(analysis_row)

        if analysis_results:
            analysis_df = pd.DataFrame(analysis_results)
            logger.info("\n" + "#" * 70)
            logger.info("QUANTIZATION IMPACT ANALYSIS")
            logger.info("#" * 70)
            logger.info("\n" + analysis_df.round(3).to_string(index=False))
            analysis_path = self.output_dir / "quantization_impact.csv"
            analysis_df.to_csv(analysis_path, index=False)
            logger.info(f"Analysis saved: {analysis_path}")

    def generate_all_visualizations(self, results_df: pd.DataFrame):
        """Generate all visualization plots as separate files."""
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
            }
        )

        # Generate individual plots
        self._plot_latency_vs_memory(results_df)
        self._plot_energy_vs_accuracy(results_df)
        self._plot_throughput_comparison(results_df)
        self._plot_accuracy_comparison(results_df)
        self._plot_pareto_frontier_latency_accuracy(results_df)
        self._plot_pareto_frontier_energy_accuracy(results_df)
        self._plot_pareto_frontier_size_accuracy(results_df)
        self._plot_speedup_by_model(results_df)
        self._plot_task_accuracy_degradation(results_df)
        self._plot_energy_consumption_breakdown(results_df)
        self._plot_latency_distribution(results_df)
        self._plot_accuracy_heatmap(results_df)
        self._plot_efficiency_radar(results_df)
        self._plot_power_consumption(results_df)

        logger.info(f"All visualizations saved to: {self.plots_dir}")

    def _plot_latency_vs_memory(self, results_df: pd.DataFrame):
        """Generate latency vs memory scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}
        markers = {"fp16": "o", "int8": "s", "int4": "^"}

        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax.scatter(
                data["mean_latency_s"] * 1000,  # Convert to ms
                data["mean_peak_mem_mb"] / 1024,  # Convert to GB
                label=str(precision).upper(),
                s=200,
                alpha=0.7,
                c=colors.get(precision, "#95a5a6"),
                marker=markers.get(precision, "o"),
                edgecolors="black",
                linewidths=1,
            )

        # Add model labels
        for _, row in results_df.iterrows():
            model_name = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            ax.annotate(
                model_name,
                (row["mean_latency_s"] * 1000, row["mean_peak_mem_mb"] / 1024),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_title("Latency vs Memory Trade-off")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "latency_vs_memory.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "latency_vs_memory.pdf", bbox_inches="tight")
        plt.close()

    def _plot_energy_vs_accuracy(self, results_df: pd.DataFrame):
        """Generate energy vs accuracy scatter plot."""
        if "energy_kwh" not in results_df.columns:
            return

        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if not accuracy_cols:
            return

        results_df = results_df.copy()
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}
        markers = {"fp16": "o", "int8": "s", "int4": "^"}

        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax.scatter(
                data["energy_kwh"] * 1000,  # Convert to Wh
                data["avg_accuracy"] * 100,  # Convert to percentage
                label=str(precision).upper(),
                s=200,
                alpha=0.7,
                c=colors.get(precision, "#95a5a6"),
                marker=markers.get(precision, "o"),
                edgecolors="black",
                linewidths=1,
            )

        # Add model labels
        for _, row in results_df.iterrows():
            model_name = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            ax.annotate(
                model_name,
                (row["energy_kwh"] * 1000, row["avg_accuracy"] * 100),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_xlabel("Energy Consumption (Wh)")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title("Energy vs Accuracy Trade-off")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "energy_vs_accuracy.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "energy_vs_accuracy.pdf", bbox_inches="tight")
        plt.close()

    def _plot_throughput_comparison(self, results_df: pd.DataFrame):
        """Generate throughput comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 7))

        unique_models = results_df["model"].unique()
        models_short = [
            str(m).split("/")[-1].replace("-Instruct", "") for m in unique_models
        ]
        x_positions = np.arange(len(unique_models))
        bar_width = 0.25
        precisions = sorted(results_df["precision"].unique())
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, precision in enumerate(precisions):
            throughputs = []
            for model in unique_models:
                model_data = results_df[
                    (results_df["model"] == model)
                    & (results_df["precision"] == precision)
                ]
                if len(model_data) > 0:
                    throughputs.append(model_data["tokens_per_second"].values[0])
                else:
                    throughputs.append(0)

            positions = x_positions + i * bar_width
            ax.bar(
                positions,
                throughputs,
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Throughput (Tokens/Second)")
        ax.set_title("Throughput Comparison Across Models and Precisions")
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(models_short, rotation=45, ha="right")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "throughput_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "throughput_comparison.pdf", bbox_inches="tight")
        plt.close()

    def _plot_accuracy_comparison(self, results_df: pd.DataFrame):
        """Generate accuracy comparison bar chart."""
        accuracy_cols = [col for col in results_df.columns if col.endswith("_accuracy")]
        has_valid_accuracy = (
            accuracy_cols
            and not results_df[accuracy_cols].isna().all().all()
            and results_df[accuracy_cols].max().max() > 0
        )

        if not has_valid_accuracy:
            return

        results_df = results_df.copy()
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(12, 7))

        unique_models = results_df["model"].unique()
        models_short = [
            str(m).split("/")[-1].replace("-Instruct", "") for m in unique_models
        ]
        x_positions = np.arange(len(unique_models))
        bar_width = 0.25
        precisions = sorted(results_df["precision"].unique())
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, precision in enumerate(precisions):
            accuracies = []
            for model in unique_models:
                model_data = results_df[
                    (results_df["model"] == model)
                    & (results_df["precision"] == precision)
                ]
                if len(model_data) > 0:
                    accuracies.append(model_data["avg_accuracy"].values[0] * 100)
                else:
                    accuracies.append(0)

            positions = x_positions + i * bar_width
            ax.bar(
                positions,
                accuracies,
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title("Average Accuracy Across Benchmarks")
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(models_short, rotation=45, ha="right")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "accuracy_comparison.pdf", bbox_inches="tight")
        plt.close()

    def _plot_pareto_frontier_latency_accuracy(self, results_df: pd.DataFrame):
        """Generate Pareto frontier for latency vs accuracy."""
        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if not accuracy_cols:
            return

        results_df = results_df.copy()
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}
        markers = {"fp16": "o", "int8": "s", "int4": "^"}

        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax.scatter(
                data["mean_latency_s"] * 1000,
                data["avg_accuracy"] * 100,
                label=str(precision).upper(),
                s=200,
                alpha=0.7,
                c=colors.get(precision, "#95a5a6"),
                marker=markers.get(precision, "o"),
                edgecolors="black",
                linewidths=1,
            )

        # Compute and plot Pareto frontier
        latencies = results_df["mean_latency_s"].values * 1000
        accuracies = results_df["avg_accuracy"].values * 100

        # Find Pareto optimal points (minimize latency, maximize accuracy)
        pareto_points = []
        for i in range(len(latencies)):
            is_pareto = True
            for j in range(len(latencies)):
                if i != j:
                    # Check if point j dominates point i
                    if latencies[j] <= latencies[i] and accuracies[j] >= accuracies[i]:
                        if latencies[j] < latencies[i] or accuracies[j] > accuracies[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_points.append((latencies[i], accuracies[i]))

        if pareto_points:
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(
                pareto_x,
                pareto_y,
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Pareto Frontier",
            )

        # Add model labels
        for _, row in results_df.iterrows():
            model_name = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            ax.annotate(
                model_name,
                (row["mean_latency_s"] * 1000, row["avg_accuracy"] * 100),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title("Pareto Frontier: Latency vs Accuracy")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "pareto_latency_accuracy.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "pareto_latency_accuracy.pdf", bbox_inches="tight")
        plt.close()

    def _plot_pareto_frontier_energy_accuracy(self, results_df: pd.DataFrame):
        """Generate Pareto frontier for energy vs accuracy."""
        if "energy_kwh" not in results_df.columns:
            return

        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if not accuracy_cols:
            return

        results_df = results_df.copy()
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}
        markers = {"fp16": "o", "int8": "s", "int4": "^"}

        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax.scatter(
                data["energy_kwh"] * 1000,
                data["avg_accuracy"] * 100,
                label=str(precision).upper(),
                s=200,
                alpha=0.7,
                c=colors.get(precision, "#95a5a6"),
                marker=markers.get(precision, "o"),
                edgecolors="black",
                linewidths=1,
            )

        # Compute and plot Pareto frontier
        energies = results_df["energy_kwh"].values * 1000
        accuracies = results_df["avg_accuracy"].values * 100

        pareto_points = []
        for i in range(len(energies)):
            is_pareto = True
            for j in range(len(energies)):
                if i != j:
                    if energies[j] <= energies[i] and accuracies[j] >= accuracies[i]:
                        if energies[j] < energies[i] or accuracies[j] > accuracies[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_points.append((energies[i], accuracies[i]))

        if pareto_points:
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(
                pareto_x,
                pareto_y,
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Pareto Frontier",
            )

        for _, row in results_df.iterrows():
            model_name = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            ax.annotate(
                model_name,
                (row["energy_kwh"] * 1000, row["avg_accuracy"] * 100),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_xlabel("Energy Consumption (Wh)")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title("Pareto Frontier: Energy vs Accuracy")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "pareto_energy_accuracy.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "pareto_energy_accuracy.pdf", bbox_inches="tight")
        plt.close()

    def _plot_pareto_frontier_size_accuracy(self, results_df: pd.DataFrame):
        """Generate Pareto frontier for model size vs accuracy."""
        if "model_size_gb" not in results_df.columns:
            return

        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if not accuracy_cols:
            return

        results_df = results_df.copy()
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}
        markers = {"fp16": "o", "int8": "s", "int4": "^"}

        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax.scatter(
                data["model_size_gb"],
                data["avg_accuracy"] * 100,
                label=str(precision).upper(),
                s=200,
                alpha=0.7,
                c=colors.get(precision, "#95a5a6"),
                marker=markers.get(precision, "o"),
                edgecolors="black",
                linewidths=1,
            )

        # Compute and plot Pareto frontier
        sizes = results_df["model_size_gb"].values
        accuracies = results_df["avg_accuracy"].values * 100

        # Find Pareto optimal points (minimize size, maximize accuracy)
        pareto_points = []
        for i in range(len(sizes)):
            is_pareto = True
            for j in range(len(sizes)):
                if i != j:
                    # Check if point j dominates point i
                    if sizes[j] <= sizes[i] and accuracies[j] >= accuracies[i]:
                        if sizes[j] < sizes[i] or accuracies[j] > accuracies[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_points.append((sizes[i], accuracies[i]))

        if pareto_points:
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(
                pareto_x,
                pareto_y,
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Pareto Frontier",
            )

        for _, row in results_df.iterrows():
            model_name = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            ax.annotate(
                model_name,
                (row["model_size_gb"], row["avg_accuracy"] * 100),
                fontsize=8,
                alpha=0.8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_xlabel("Model Size (GB)")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title("Pareto Frontier: Model Size vs Accuracy")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "pareto_size_accuracy.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "pareto_size_accuracy.pdf", bbox_inches="tight")
        plt.close()

    def _plot_speedup_by_model(self, results_df: pd.DataFrame):
        """Generate speedup comparison by model."""
        speedup_data = []

        for model in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model]
            fp16_data = model_data[model_data["precision"] == "fp16"]
            if len(fp16_data) == 0:
                continue

            fp16_latency = fp16_data["mean_latency_s"].values[0]
            model_short = str(model).split("/")[-1].replace("-Instruct", "")

            for _, row in model_data.iterrows():
                speedup = fp16_latency / row["mean_latency_s"]
                speedup_data.append(
                    {
                        "model": model_short,
                        "precision": row["precision"],
                        "speedup": speedup,
                    }
                )

        if not speedup_data:
            return

        speedup_df = pd.DataFrame(speedup_data)

        fig, ax = plt.subplots(figsize=(12, 7))

        unique_models = speedup_df["model"].unique()
        x_positions = np.arange(len(unique_models))
        bar_width = 0.25
        precisions = sorted(speedup_df["precision"].unique())
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, precision in enumerate(precisions):
            speedups = []
            for model in unique_models:
                model_data = speedup_df[
                    (speedup_df["model"] == model)
                    & (speedup_df["precision"] == precision)
                ]
                if len(model_data) > 0:
                    speedups.append(model_data["speedup"].values[0])
                else:
                    speedups.append(0)

            positions = x_positions + i * bar_width
            ax.bar(
                positions,
                speedups,
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Baseline (FP16)",
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Speedup (relative to FP16)")
        ax.set_title("Quantization Speedup by Model")
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(unique_models, rotation=45, ha="right")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "speedup_by_model.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "speedup_by_model.pdf", bbox_inches="tight")
        plt.close()

    def _plot_task_accuracy_degradation(self, results_df: pd.DataFrame):
        """Generate task-specific accuracy degradation plot."""
        degradation_data = []

        for model in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model]
            fp16_data = model_data[model_data["precision"] == "fp16"]
            if len(fp16_data) == 0:
                continue

            model_short = str(model).split("/")[-1].replace("-Instruct", "")

            for _, row in model_data.iterrows():
                if row["precision"] == "fp16":
                    continue

                for task in self.accuracy_tasks:
                    acc_col = f"{task}_accuracy"
                    if acc_col in fp16_data.columns and acc_col in row:
                        fp16_acc = fp16_data[acc_col].values[0]
                        quant_acc = row[acc_col]
                        if fp16_acc > 0 and not pd.isna(quant_acc):
                            degradation = ((fp16_acc - quant_acc) / fp16_acc) * 100
                            degradation_data.append(
                                {
                                    "model": model_short,
                                    "precision": row["precision"],
                                    "task": task.upper(),
                                    "degradation": degradation,
                                }
                            )

        if not degradation_data:
            return

        degradation_df = pd.DataFrame(degradation_data)

        fig, ax = plt.subplots(figsize=(14, 7))

        tasks = degradation_df["task"].unique()
        precisions = sorted(degradation_df["precision"].unique())
        x_positions = np.arange(len(tasks))
        bar_width = 0.35
        colors = {"int8": "#3498db", "int4": "#e74c3c"}

        offset = 0
        for i, precision in enumerate(precisions):
            if precision == "fp16":
                continue
            avg_degradations = []
            std_degradations = []
            for task in tasks:
                task_data = degradation_df[
                    (degradation_df["task"] == task)
                    & (degradation_df["precision"] == precision)
                ]
                avg_degradations.append(task_data["degradation"].mean())
                std_degradations.append(task_data["degradation"].std())

            positions = x_positions + (offset - 0.5) * bar_width
            ax.bar(
                positions,
                avg_degradations,
                width=bar_width,
                yerr=std_degradations,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
                capsize=5,
            )
            offset += 1

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
        ax.set_xlabel("Benchmark Task")
        ax.set_ylabel("Accuracy Degradation (%)")
        ax.set_title("Task-Specific Accuracy Degradation from Quantization")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tasks)
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "task_accuracy_degradation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.plots_dir / "task_accuracy_degradation.pdf", bbox_inches="tight"
        )
        plt.close()

    def _plot_energy_consumption_breakdown(self, results_df: pd.DataFrame):
        """Generate energy consumption breakdown by model."""
        if "energy_kwh" not in results_df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        unique_models = results_df["model"].unique()
        models_short = [
            str(m).split("/")[-1].replace("-Instruct", "") for m in unique_models
        ]
        x_positions = np.arange(len(unique_models))
        bar_width = 0.25
        precisions = sorted(results_df["precision"].unique())
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, precision in enumerate(precisions):
            energies = []
            for model in unique_models:
                model_data = results_df[
                    (results_df["model"] == model)
                    & (results_df["precision"] == precision)
                ]
                if len(model_data) > 0:
                    energies.append(
                        model_data["energy_kwh"].values[0] * 1000
                    )  # Convert to Wh
                else:
                    energies.append(0)

            positions = x_positions + i * bar_width
            ax.bar(
                positions,
                energies,
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Energy Consumption (Wh)")
        ax.set_title("Energy Consumption by Model and Precision")
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(models_short, rotation=45, ha="right")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "energy_consumption.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "energy_consumption.pdf", bbox_inches="tight")
        plt.close()

    def _plot_latency_distribution(self, results_df: pd.DataFrame):
        """Generate latency distribution comparison."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Prepare data for visualization
        latency_data = []

        for _, row in results_df.iterrows():
            model_short = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            label = f"{model_short}\n({row['precision'].upper()})"

            mean = row["mean_latency_s"] * 1000
            std = row.get("std_latency_s", 0) * 1000

            latency_data.append(
                {
                    "label": label,
                    "mean": mean,
                    "std": std,
                    "precision": row["precision"],
                }
            )

        # Sort by mean latency
        latency_data.sort(key=lambda x: x["mean"])

        x_positions = np.arange(len(latency_data))
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, data in enumerate(latency_data):
            color = colors.get(data["precision"], "#95a5a6")
            ax.bar(
                i,
                data["mean"],
                width=0.6,
                alpha=0.8,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.errorbar(
                i,
                data["mean"],
                yerr=data["std"],
                fmt="none",
                color="black",
                capsize=5,
                capthick=1.5,
            )

        ax.set_xlabel("Model Configuration")
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title("Latency Comparison Across Configurations")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [d["label"] for d in latency_data], rotation=45, ha="right", fontsize=8
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors[p], label=p.upper(), alpha=0.8)
            for p in ["fp16", "int8", "int4"]
            if p in colors
        ]
        ax.legend(handles=legend_elements, title="Precision")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "latency_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "latency_distribution.pdf", bbox_inches="tight")
        plt.close()

    def _plot_accuracy_heatmap(self, results_df: pd.DataFrame):
        """Generate accuracy heatmap across models, precisions, and tasks."""
        accuracy_cols = [col for col in results_df.columns if col.endswith("_accuracy")]
        if not accuracy_cols:
            return

        # Prepare data for heatmap
        heatmap_data = []

        for _, row in results_df.iterrows():
            model_short = str(row["model"]).split("/")[-1].replace("-Instruct", "")
            config = f"{model_short}\n({row['precision'].upper()})"

            for col in accuracy_cols:
                task = col.replace("_accuracy", "").upper()
                accuracy = row[col] * 100 if not pd.isna(row[col]) else 0
                heatmap_data.append(
                    {"config": config, "task": task, "accuracy": accuracy}
                )

        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(index="config", columns="task", values="accuracy")

        fig, ax = plt.subplots(figsize=(10, 12))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Accuracy (%)"},
            linewidths=0.5,
            vmin=40,
            vmax=80,
        )

        ax.set_title("Accuracy Heatmap: Models × Tasks")
        ax.set_xlabel("Benchmark Task")
        ax.set_ylabel("Model Configuration")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "accuracy_heatmap.pdf", bbox_inches="tight")
        plt.close()

    def _plot_efficiency_radar(self, results_df: pd.DataFrame):
        """Generate efficiency radar/spider chart for each precision."""
        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if not accuracy_cols or "energy_kwh" not in results_df.columns:
            return

        df_norm = results_df.copy()
        df_norm["avg_accuracy"] = df_norm[accuracy_cols].mean(axis=1)

        # Normalize metrics (higher is better for all)
        df_norm["accuracy_norm"] = (
            df_norm["avg_accuracy"] / df_norm["avg_accuracy"].max()
        )
        df_norm["throughput_norm"] = (
            df_norm["tokens_per_second"] / df_norm["tokens_per_second"].max()
        )
        df_norm["latency_norm"] = (
            df_norm["mean_latency_s"].min() / df_norm["mean_latency_s"]
        )
        df_norm["energy_norm"] = df_norm["energy_kwh"].min() / df_norm["energy_kwh"]
        df_norm["memory_norm"] = (
            df_norm["mean_peak_mem_mb"].min() / df_norm["mean_peak_mem_mb"]
        )

        categories = [
            "Accuracy",
            "Throughput",
            "Latency\nEfficiency",
            "Energy\nEfficiency",
            "Memory\nEfficiency",
        ]
        num_vars = len(categories)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

        precisions = ["fp16", "int8", "int4"]

        for ax, precision in zip(axes, precisions):
            precision_data = df_norm[df_norm["precision"] == precision]

            angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
            angles += angles[:1]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9)

            for _, row in precision_data.iterrows():
                model_short = str(row["model"]).split("/")[-1].replace("-Instruct", "")
                values = [
                    row["accuracy_norm"],
                    row["throughput_norm"],
                    row["latency_norm"],
                    row["energy_norm"],
                    row["memory_norm"],
                ]
                values += values[:1]

                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    linestyle="solid",
                    label=model_short,
                    alpha=0.8,
                )
                ax.fill(angles, values, alpha=0.1)

            ax.set_title(f"{precision.upper()}", size=14, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "efficiency_radar.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "efficiency_radar.pdf", bbox_inches="tight")
        plt.close()

    def _plot_power_consumption(self, results_df: pd.DataFrame):
        """Generate power consumption comparison."""
        if "avg_power_watts" not in results_df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        unique_models = results_df["model"].unique()
        models_short = [
            str(m).split("/")[-1].replace("-Instruct", "") for m in unique_models
        ]
        x_positions = np.arange(len(unique_models))
        bar_width = 0.25
        precisions = sorted(results_df["precision"].unique())
        colors = {"fp16": "#2ecc71", "int8": "#3498db", "int4": "#e74c3c"}

        for i, precision in enumerate(precisions):
            powers = []
            errors = []
            for model in unique_models:
                model_data = results_df[
                    (results_df["model"] == model)
                    & (results_df["precision"] == precision)
                ]
                if len(model_data) > 0:
                    powers.append(model_data["avg_power_watts"].values[0])
                    std_col = "std_power_watts"
                    if std_col in model_data.columns:
                        errors.append(model_data[std_col].values[0])
                    else:
                        errors.append(0)
                else:
                    powers.append(0)
                    errors.append(0)

            positions = x_positions + i * bar_width
            ax.bar(
                positions,
                powers,
                width=bar_width,
                yerr=errors,
                label=str(precision).upper(),
                alpha=0.8,
                color=colors.get(precision, "#95a5a6"),
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Average Power (Watts)")
        ax.set_title("Power Consumption by Model and Precision")
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(models_short, rotation=45, ha="right")
        ax.legend(title="Precision")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "power_consumption.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.plots_dir / "power_consumption.pdf", bbox_inches="tight")
        plt.close()

    def generate_summary_statistics(self, results_df: pd.DataFrame):
        """Generate summary statistics by precision."""
        summary_stats = (
            results_df.groupby(["precision"])
            .agg(
                {
                    "mean_latency_s": ["mean", "std", "min", "max"],
                    "mean_peak_mem_mb": ["mean", "std", "min", "max"],
                    "tokens_per_second": ["mean", "std", "min", "max"],
                }
            )
            .round(4)
        )

        if "energy_kwh" in results_df.columns:
            energy_stats = (
                results_df.groupby(["precision"])
                .agg(
                    {"energy_kwh": ["mean", "std"], "avg_power_watts": ["mean", "std"]}
                )
                .round(4)
            )
            summary_stats = pd.concat([summary_stats, energy_stats], axis=1)

        # Flatten multi-level column headers
        summary_stats.columns = [
            "_".join(col).strip() for col in summary_stats.columns.values
        ]

        logger.info("\n" + "#" * 70)
        logger.info("SUMMARY STATISTICS BY PRECISION")
        logger.info("#" * 70)
        logger.info("\n" + str(summary_stats))
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_path)
        logger.info(f"Summary saved: {summary_path}")

    def generate_results_table(self, results_df: pd.DataFrame):
        """Generate results table for publication."""
        results_columns = [
            "model",
            "num_parameters_b",
            "precision",
            "model_size_gb",
            "mean_latency_s",
            "mean_peak_mem_mb",
            "tokens_per_second",
        ]
        if "energy_kwh" in results_df.columns:
            results_columns.append("energy_kwh")
        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if accuracy_cols:
            results_columns.extend(accuracy_cols[:3])

        # Filter to only include columns that exist
        results_columns = [col for col in results_columns if col in results_df.columns]

        results_table = results_df[results_columns].copy()
        results_table["model"] = (
            results_table["model"].astype(str).str.split("/").str[-1]
        )
        rename_map = {
            "num_parameters_b": "Params (B)",
            "model_size_gb": "Size (GB)",
            "mean_latency_s": "Latency (s)",
            "mean_peak_mem_mb": "Memory (MB)",
            "tokens_per_second": "Tokens/s",
            "energy_kwh": "Energy (kWh)",
        }
        results_table = results_table.rename(columns=rename_map).round(4)

        logger.info("\n" + "#" * 70)
        logger.info("RESULTS TABLE")
        logger.info("#" * 70)
        logger.info("\n" + results_table.to_string(index=False))

        csv_path = self.output_dir / "results_table.csv"
        results_table.to_csv(csv_path, index=False)
        latex_table = results_table.to_latex(index=False, float_format="%.4f")
        latex_path = self.output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        logger.info(f"Results table saved: {csv_path}\nLaTeX table saved: {latex_path}")

    def generate_executive_summary(self, results_df: pd.DataFrame):
        """Generate executive summary report."""
        report_lines = [
            "=" * 70,
            "SLiM-EVAL: EXECUTIVE SUMMARY REPORT",
            "=" * 70,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nModels Evaluated: {len(results_df['model'].unique())}",
            f"Precision Modes: {', '.join(results_df['precision'].unique())}",
            f"Total Configurations: {len(results_df)}",
            "\n" + "=" * 70,
            "\nKEY FINDINGS:",
            "=" * 70,
        ]

        if len(results_df) > 0:
            # Fastest model
            if "mean_latency_s" in results_df.columns:
                lat_series = results_df["mean_latency_s"].dropna()
                if not lat_series.empty:
                    fastest_idx = lat_series.idxmin()
                    fastest = results_df.loc[fastest_idx]
                    report_lines.append("\n1. FASTEST MODEL:")
                    report_lines.append(
                        f"   {str(fastest['model']).split('/')[-1]} ({fastest['precision']})"
                    )
                    report_lines.append(
                        f"   Latency: {fastest['mean_latency_s'] * 1000:.2f} ms"
                    )

            # Most memory efficient
            if "mean_peak_mem_mb" in results_df.columns:
                mem_series = results_df["mean_peak_mem_mb"].dropna()
                if not mem_series.empty:
                    mem_idx = mem_series.idxmin()
                    mem_efficient = results_df.loc[mem_idx]
                    report_lines.append("\n2. MOST MEMORY EFFICIENT:")
                    report_lines.append(
                        f"   {str(mem_efficient['model']).split('/')[-1]} ({mem_efficient['precision']})"
                    )
                    report_lines.append(
                        f"   Memory: {mem_efficient['mean_peak_mem_mb'] / 1024:.2f} GB"
                    )

            # Highest throughput
            if "tokens_per_second" in results_df.columns:
                tps_series = results_df["tokens_per_second"].dropna()
                if not tps_series.empty:
                    tps_idx = tps_series.idxmax()
                    highest_throughput = results_df.loc[tps_idx]
                    report_lines.append("\n3. HIGHEST THROUGHPUT:")
                    report_lines.append(
                        f"   {str(highest_throughput['model']).split('/')[-1]} ({highest_throughput['precision']})"
                    )
                    report_lines.append(
                        f"   Throughput: {highest_throughput['tokens_per_second']:.2f} tokens/s"
                    )

            # Best accuracy
            accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
            if accuracy_cols:
                avg_acc_series = results_df[accuracy_cols].mean(axis=1, skipna=True)
                valid_avg = avg_acc_series.dropna()
                if not valid_avg.empty and valid_avg.max() > 0:
                    best_idx = valid_avg.idxmax()
                    best_accuracy = results_df.loc[best_idx]
                    report_lines.append("\n4. BEST AVERAGE ACCURACY:")
                    report_lines.append(
                        f"   {str(best_accuracy['model']).split('/')[-1]} ({best_accuracy['precision']})"
                    )
                    report_lines.append(
                        f"   Avg Accuracy: {valid_avg.loc[best_idx] * 100:.2f}%"
                    )

            # Most energy efficient
            if "energy_kwh" in results_df.columns:
                energy_series = results_df["energy_kwh"].dropna()
                if not energy_series.empty:
                    energy_idx = energy_series.idxmin()
                    energy_efficient = results_df.loc[energy_idx]
                    report_lines.append("\n5. MOST ENERGY EFFICIENT:")
                    report_lines.append(
                        f"   {str(energy_efficient['model']).split('/')[-1]} ({energy_efficient['precision']})"
                    )
                    report_lines.append(
                        f"   Energy: {energy_efficient['energy_kwh'] * 1000:.2f} Wh"
                    )

            # Best balanced (efficiency score: accuracy / latency)
            if accuracy_cols and "mean_latency_s" in results_df.columns:
                results_df_copy = results_df.copy()
                results_df_copy["avg_accuracy"] = results_df_copy[accuracy_cols].mean(
                    axis=1
                )
                results_df_copy["efficiency_score"] = (
                    results_df_copy["avg_accuracy"] / results_df_copy["mean_latency_s"]
                )
                eff_series = results_df_copy["efficiency_score"].dropna()
                if not eff_series.empty:
                    eff_idx = eff_series.idxmax()
                    balanced = results_df_copy.loc[eff_idx]
                    report_lines.append("\n6. BEST BALANCED (Efficiency Score):")
                    report_lines.append(
                        f"   {str(balanced['model']).split('/')[-1]} ({balanced['precision']})"
                    )
                    report_lines.append(
                        f"   Latency: {balanced['mean_latency_s'] * 1000:.2f} ms, Accuracy: {balanced['avg_accuracy'] * 100:.2f}%"
                    )

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)
        logger.info("\n" + report_text)
        report_path = self.output_dir / "executive_summary.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Executive summary saved: {report_path}")

    def export_json(self, results_df: pd.DataFrame):
        """Export results to JSON format."""
        json_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_configurations": len(results_df),
            },
            "results": results_df.to_dict(orient="records"),
        }
        json_path = self.output_dir / "complete_results.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"JSON results saved: {json_path}")
