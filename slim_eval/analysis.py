"""Analysis and visualization utilities for SLiM-Eval."""

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes and visualizes benchmark results."""

    def __init__(self, output_dir: Path, accuracy_tasks: list):
        """Initialize results analyzer.

        Args:
            output_dir: Directory for saving analysis outputs.
            accuracy_tasks: List of accuracy task names.
        """
        self.output_dir = output_dir
        self.accuracy_tasks = accuracy_tasks

    def load_results_from_json(self) -> pd.DataFrame:
        """Load all benchmark results from JSON files in model directories.

        Returns:
            DataFrame with all results combined.
        """
        all_results = []

        # Find all model directories (first level)
        for model_dir in self.output_dir.iterdir():
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

        self.analyze_quantization_impact(results_df)
        self.generate_visualizations(results_df)
        self.generate_summary_statistics(results_df)
        self.generate_paper_table(results_df)
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

    def generate_visualizations(self, results_df: pd.DataFrame):
        """Generate visualization plots."""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Latency vs Memory scatter plot
        ax1 = axes[0, 0]
        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax1.scatter(
                data["mean_latency_s"],
                data["mean_peak_mem_mb"],
                label=str(precision).upper(),
                s=150,
                alpha=0.7,
            )
        # Add model labels
        for _, row in results_df.iterrows():
            model_name = str(row['model']).split('/')[-1]
            label = f"{model_name}\n{row['precision']}"
            ax1.annotate(
                label,
                (row['mean_latency_s'], row['mean_peak_mem_mb']),
                fontsize=7,
                alpha=0.8,
                ha='center'
            )
        ax1.set_xlabel("Latency (seconds)")
        ax1.set_ylabel("Peak Memory (MB)")
        ax1.set_title("Latency vs Memory")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy vs Accuracy scatter plot
        if "energy_kwh" in results_df.columns:
            ax2 = axes[0, 1]
            acc_col = [col for col in results_df.columns if "accuracy" in col]
            if acc_col:
                acc_col = acc_col[0]
                for precision in results_df["precision"].unique():
                    data = results_df[results_df["precision"] == precision]
                    ax2.scatter(
                        data["energy_kwh"],
                        data[acc_col],
                        label=str(precision).upper(),
                        s=150,
                        alpha=0.7,
                    )
                # Add model labels
                for _, row in results_df.iterrows():
                    if acc_col in row and not pd.isna(row[acc_col]):
                        model_name = str(row['model']).split('/')[-1]
                        label = f"{model_name}\n{row['precision']}"
                        ax2.annotate(
                            label,
                            (row['energy_kwh'], row[acc_col]),
                            fontsize=7,
                            alpha=0.8,
                            ha='center'
                        )
                ax2.set_xlabel("Energy (kWh)")
                ax2.set_ylabel("Accuracy")
                ax2.set_title("Energy vs Accuracy Trade-off")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # Throughput bar chart
        ax3 = axes[1, 0]
        unique_models = results_df["model"].unique()
        models_short = [str(m).split("/")[-1] for m in unique_models]
        x_positions = range(len(unique_models))
        bar_width = 0.25
        precisions = sorted(results_df["precision"].unique())

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

            positions = [x + i * bar_width for x in x_positions]
            ax3.bar(
                positions,
                throughputs,
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.7,
            )
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Tokens/Second")
        ax3.set_title("Throughput by Model and Precision")
        ax3.set_xticks([p + bar_width for p in x_positions])
        ax3.set_xticklabels(models_short, rotation=45, ha="right")
        ax3.legend()

        # Accuracy bar chart
        ax4 = axes[1, 1]
        accuracy_cols = [col for col in results_df.columns if col.endswith("_accuracy")]
        has_valid_accuracy = (
            accuracy_cols
            and not results_df[accuracy_cols].isna().all().all()
            and results_df[accuracy_cols].max().max() > 0
        )
        if has_valid_accuracy:
            results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)

            for i, precision in enumerate(precisions):
                accuracies = []
                for model in unique_models:
                    model_data = results_df[
                        (results_df["model"] == model)
                        & (results_df["precision"] == precision)
                    ]
                    if len(model_data) > 0:
                        accuracies.append(model_data["avg_accuracy"].values[0])
                    else:
                        accuracies.append(0)

                positions = [x + i * bar_width for x in x_positions]
                ax4.bar(
                    positions,
                    accuracies,
                    width=bar_width,
                    label=str(precision).upper(),
                    alpha=0.7,
                )
            ax4.set_xlabel("Model")
            ax4.set_ylabel("Average Accuracy")
            ax4.set_title("Average Accuracy Across Tasks")
            ax4.set_xticks([p + bar_width for p in x_positions])
            ax4.set_xticklabels(models_short, rotation=45, ha="right")
            ax4.legend()
        else:
            ax4.text(
                0.5, 0.5, "No Accuracy Data", ha="center", va="center", fontsize=14
            )
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()
        plot_path = self.output_dir / "pareto_frontiers.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Visualizations saved: {plot_path}")

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
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

        logger.info("\n" + "#" * 70)
        logger.info("SUMMARY STATISTICS BY PRECISION")
        logger.info("#" * 70)
        logger.info("\n" + str(summary_stats))
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_path)
        logger.info(f"Summary saved: {summary_path}")

    def generate_paper_table(self, results_df: pd.DataFrame):
        """Generate publication-ready results table."""
        paper_columns = [
            "model",
            "num_parameters_b",
            "precision",
            "mean_latency_s",
            "mean_peak_mem_mb",
            "tokens_per_second",
        ]
        if "energy_kwh" in results_df.columns:
            paper_columns.append("energy_kwh")
        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if accuracy_cols:
            paper_columns.extend(accuracy_cols[:3])

        paper_table = results_df[paper_columns].copy()
        paper_table["model"] = paper_table["model"].astype(str).str.split("/").str[-1]
        rename_map = {
            "num_parameters_b": "Params (B)",
            "mean_latency_s": "Latency (s)",
            "mean_peak_mem_mb": "Memory (MB)",
            "tokens_per_second": "Tokens/s",
            "energy_kwh": "Energy (kWh)",
        }
        paper_table = paper_table.rename(columns=rename_map).round(4)

        logger.info("\n" + "#" * 70)
        logger.info("PAPER-READY RESULTS TABLE")
        logger.info("#" * 70)
        logger.info("\n" + paper_table.to_string(index=False))

        paper_csv_path = self.output_dir / "paper_results.csv"
        paper_table.to_csv(paper_csv_path, index=False)
        latex_table = paper_table.to_latex(index=False, float_format="%.4f")
        latex_path = self.output_dir / "paper_results.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        logger.info(
            f"Paper table saved: {paper_csv_path}\nLaTeX table saved: {latex_path}"
        )

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
                    report_lines.append(f"   Latency: {fastest['mean_latency_s']:.4f}s")

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
                        f"   Memory: {mem_efficient['mean_peak_mem_mb']:.2f} MB"
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
                        f"   Avg Accuracy: {valid_avg.loc[best_idx]:.4f}"
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
