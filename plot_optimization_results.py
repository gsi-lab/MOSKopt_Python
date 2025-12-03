"""
Optimization Results Plotting Tool

A simple plotting tool for MOSKopt optimization results that supports:
- Both deterministic and stochastic optimization results
- CSV file format
- Smart filtering of failed simulations and restart points
- Professional quality plots with proper formatting

Usage:
    python plot_optimization_results.py                    # Uses CSV_FILE_PATH from config
    python plot_optimization_results.py --file results.csv # Use specific file
    python plot_optimization_results.py --raw              # No filtering (raw data)
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION: Just paste your CSV file path here and run the script!
# ============================================================================
CSV_FILE_PATH = r"C:\Users\tusas\OneDrive - Danmarks Tekniske Universitet\Skrivebord\CursorCodes\Paper1codes\MOSKopt_Python\optimization_results\stochastic_adaptive_progress_csv_20251203_004800.csv"  # <-- Paste your CSV file path between the quotes

# Example: CSV_FILE_PATH = r"C:\Users\...\optimization_results\stochastic_adaptive_progress_csv_20251203_004800.csv"
# Leave empty to use --file argument: CSV_FILE_PATH = r""
# ============================================================================


def plot_optimization_results(
    results_file,
    save_dir="optimization_results",
    num_seed_points: int = 25,
    allow_filter: bool = True,
):
    """
    Plot optimization progress from CSV results file.
    Supports both deterministic and stochastic optimization results.

    CSV input: expects columns 'Iteration' and 'LCO_Objective'. The first
    `num_seed_points` rows are treated as seed points.
    """

    # Load results from CSV
    df = pd.read_csv(results_file)
    print(f"CSV columns found: {list(df.columns)}")

    # Check for objective column (try different possible names)
    objective_col = None
    possible_names = ["Objective", "LCO_Objective", "objective", "LCO", "obj"]
    for col_name in possible_names:
        if col_name in df.columns:
            objective_col = col_name
            break

    if objective_col is None:
        raise ValueError(
            f"CSV must contain an objective column. Expected one of: {possible_names}. Found: {list(df.columns)}"
        )

    print(f"Using objective column: '{objective_col}'")

    # If filtering is disabled, use the CSV as-is (no removals or deduplication)
    if not allow_filter:
        print("Raw mode: no filtering. Plotting best-so-far across all evaluations.")

        # Read objectives as numeric array
        objectives = pd.to_numeric(df[objective_col], errors="coerce").to_numpy()
        total_points = len(objectives)

        if total_points == 0:
            raise ValueError("CSV contains no objective values to plot")

        # Determine iteration numbers (use provided Iteration column if present)
        if "Iteration" in df.columns:
            try:
                iter_nums = df["Iteration"].astype(int).tolist()
            except Exception:
                iter_nums = list(range(1, total_points + 1))
        else:
            iter_nums = list(range(1, total_points + 1))

        # Compute cumulative best-so-far across all evaluations (include seed and adaptive)
        best_so_far_all = []
        current_best = float("inf")
        for obj in objectives:
            if np.isfinite(obj) and obj < current_best:
                current_best = float(obj)
            best_so_far_all.append(current_best)

        # Mark seed/adaptive split for reporting
        if total_points > num_seed_points:
            seed_objectives = objectives[:num_seed_points]
            adaptive_objectives = objectives[num_seed_points:]
        else:
            seed_objectives = objectives
            adaptive_objectives = []

        # Create the plot showing best-so-far across all points
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(iter_nums, best_so_far_all, "g-", linewidth=3)
        # Highlight improvement points
        improvements_x = []
        improvements_y = []
        for i in range(1, len(best_so_far_all)):
            if best_so_far_all[i] < best_so_far_all[i - 1]:
                improvements_x.append(iter_nums[i])
                improvements_y.append(best_so_far_all[i])

        if improvements_x:
            ax.plot(improvements_x, improvements_y, "go", markersize=6)

        ax.set_xlabel("Iteration", fontsize=20)
        ax.set_ylabel("Best Objective Value (LCOP)", fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_title(
            "Optimization Progress (Best-So-Far Across All Evaluations)",
            fontsize=18,
            fontweight="bold",
        )

        # Set x-axis limits and ticks
        ax.set_xlim(min(iter_nums), max(iter_nums) + 1)
        try:
            step = max(1, int(len(iter_nums) / 10))
            ax.set_xticks(range(min(iter_nums), max(iter_nums) + 1, step))
        except Exception:
            pass

        # Add annotation for initial (best after seed sampling) and final values
        initial_idx = min(num_seed_points - 1, total_points - 1)
        initial_value = best_so_far_all[initial_idx]
        final_value = best_so_far_all[-1]

        annotation_text = f"Initial: {initial_value:.0f}\nFinal: {final_value:.0f}"
        ax.annotate(
            annotation_text,
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
        )

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_adaptive_progress_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Raw plot saved to: {filepath}")

        # Print a short summary and return
        print(f"Total evaluations: {total_points}")
        print(f"Improvements found: {len(improvements_x)}")
        return fig

    # Handle checkpoint files differently - they contain all evaluations with status
    elif "Status" in df.columns:
        # This is a checkpoint file - remove restart success points after failed simulations
        print(f"Checkpoint file detected. Processing {len(df)} total evaluations.")

        # Identify decision variable columns (exclude Iteration, Objective, Status)
        decision_cols = [
            col for col in df.columns if col not in ["Iteration", "Objective", "Status"]
        ]

        # Strategy: Remove all Failed entries and high objective values (penalty solutions)
        objectives_raw = pd.to_numeric(df[objective_col], errors="coerce")

        # Identify different types of points to remove
        rows_to_remove = []

        # 1. Remove all Failed entries
        failed_mask = df["Status"] == "Failed"
        failed_count = failed_mask.sum()
        if failed_count > 0:
            failed_indices = df[failed_mask].index.tolist()
            rows_to_remove.extend(failed_indices)
            print(f"Removing {failed_count} failed simulations")

        # 2. Remove high objective values (likely penalty solutions or poor restart attempts)
        penalty_threshold = 10000.0  # Appropriate threshold for ibuprofen optimization (LCO values ~6000-7000)
        high_obj_mask = objectives_raw > penalty_threshold
        high_obj_success = high_obj_mask & (df["Status"] == "Success")
        high_obj_count = high_obj_success.sum()

        if high_obj_count > 0:
            high_obj_indices = df[high_obj_success].index.tolist()
            rows_to_remove.extend(high_obj_indices)
            print(
                f"Removing {high_obj_count} successful points with high objectives (> {penalty_threshold})"
            )

            # Print some examples
            for idx in high_obj_indices[:3]:  # Show first 3 examples
                print(
                    f"  - Iteration {df.iloc[idx]['Iteration']}: obj = {objectives_raw.iloc[idx]:.1f}"
                )

        # Remove the identified points
        df_clean = df.copy()
        if rows_to_remove:
            # Remove duplicates from rows_to_remove
            rows_to_remove = list(set(rows_to_remove))
            df_clean = df_clean.drop(rows_to_remove).reset_index(drop=True)
            print(f"Removed {len(rows_to_remove)} failed/penalty/restart points total.")

        # Filter for successful evaluations only
        successful_mask = df_clean["Status"] == "Success"
        df_successful = df_clean[successful_mask].copy().reset_index(drop=True)
        print(
            f"Using {len(df_successful)} unique successful evaluations out of {len(df)} total."
        )

        # For checkpoint files, all data points are actually from the optimization process
        # The first few might be seed points, but we should use all successful evaluations
        objectives = pd.to_numeric(
            df_successful[objective_col], errors="coerce"
        ).to_numpy()
        total_points = len(objectives)

        # For checkpoint files, assume first num_seed_points are seed points
        if total_points > num_seed_points:
            seed_objectives = objectives[:num_seed_points]
            adaptive_objectives = objectives[num_seed_points:]
            # Use sequential numbering for adaptive iterations (1, 2, 3, ...) instead of actual iteration numbers
            adaptive_iteration_numbers = list(
                range(
                    num_seed_points + 1,
                    num_seed_points + 1 + len(adaptive_objectives),
                )
            )
        else:
            # Not enough points, treat all as seed points
            seed_objectives = objectives
            adaptive_objectives = []
            adaptive_iteration_numbers = []
            print(
                f"Warning: Only {total_points} successful evaluations, treating all as seed points"
            )
    else:
        # Regular results file (like your deterministic_adaptive_progress_csv)
        # Check if this is the new format with Status column or old format with Converged
        if "Status" in df.columns:
            # New format - handle like checkpoint files with status filtering
            print(
                f"Adaptive progress file with Status column detected. Processing {len(df)} total evaluations."
            )

            # Filter for successful evaluations only
            successful_mask = df["Status"] == "Success"
            df_successful = df[successful_mask].copy().reset_index(drop=True)
            print(
                f"Using {len(df_successful)} successful evaluations out of {len(df)} total."
            )

            objectives = pd.to_numeric(
                df_successful[objective_col], errors="coerce"
            ).to_numpy()
            total_points = len(objectives)
        else:
            # Old format - remove duplicate rows (restart rows) by keeping only unique combinations of decision variables
            if len(df) > 1:
                # Identify decision variable columns (exclude Iteration, Objective, Converged)
                decision_cols = [
                    col
                    for col in df.columns
                    if col not in ["Iteration", "Objective", "Converged", "Status"]
                ]

                if decision_cols:
                    # Remove duplicates based on decision variables, keeping first occurrence
                    df_unique = df.drop_duplicates(
                        subset=decision_cols, keep="first"
                    ).reset_index(drop=True)
                    print(
                        f"Removed {len(df) - len(df_unique)} duplicate/restart rows. Using {len(df_unique)} unique evaluations."
                    )
                    df = df_unique

            objectives = pd.to_numeric(df[objective_col], errors="coerce").to_numpy()
            total_points = len(objectives)

        if total_points <= num_seed_points:
            raise ValueError(
                f"CSV has only {total_points} rows, but num_seed_points is {num_seed_points}"
            )

        seed_objectives = objectives[:num_seed_points]
        adaptive_objectives = objectives[num_seed_points:]
        adaptive_iteration_numbers = (
            df["Iteration"].iloc[num_seed_points:].astype(int).tolist()
            if "Iteration" in df.columns
            else list(
                range(
                    num_seed_points + 1,
                    num_seed_points + 1 + len(adaptive_objectives),
                )
            )
        )

    # Find best feasible seed objective (filter out extreme penalty values only)
    extreme_penalty_threshold = 1e5  # Only filter out truly extreme penalty values
    feasible_seeds = seed_objectives[seed_objectives < extreme_penalty_threshold]
    if len(feasible_seeds) > 0:
        best_seed_objective = float(np.min(feasible_seeds))
    else:
        best_seed_objective = float(np.min(seed_objectives))

    print(f"Total successful points: {total_points}")
    print(
        f"Found {len(seed_objectives)} seed points and {len(adaptive_objectives)} adaptive points"
    )
    print(f"Best feasible point from seed sampling: {best_seed_objective:.3f}")

    # Calculate best-so-far for ALL adaptive points (excluding extreme penalty values)
    best_so_far = []
    current_best = best_seed_objective
    extreme_penalty_threshold = 1e5  # Only exclude truly extreme penalty values

    for obj in adaptive_objectives:
        # Only update if this is a successful simulation (not extreme penalty value)
        if np.isfinite(obj) and obj < extreme_penalty_threshold:
            if obj < current_best:
                current_best = float(obj)
        best_so_far.append(current_best)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot adaptive points (green line)
    ax.plot(adaptive_iteration_numbers, best_so_far, "g-", linewidth=3)
    ax.set_xlabel("Number of Adaptive Iterations", fontsize=20)
    ax.set_ylabel("Best Objective Value (LCOP)", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        "Optimization Progress (Adaptive Points Only)",
        fontsize=18,
        fontweight="bold",
    )

    # Set x-axis to start from num_seed_points and show every 5 iterations
    max_iter = num_seed_points + len(adaptive_objectives)
    ax.set_xlim(max(num_seed_points, num_seed_points - 1), max_iter + 5)
    ax.set_xticks(range(num_seed_points, max_iter + 6, 5))

    # Add annotation for initial and final values
    if len(best_so_far) > 0:
        initial_value = best_seed_objective  # Best seed point
        final_value = best_so_far[-1]  # Best adaptive point

        # Add annotation box in upper right corner
        annotation_text = f"Initial: {initial_value:.0f}\nFinal: {final_value:.0f}"
        ax.annotate(
            annotation_text,
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
        )

    plt.tight_layout()
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_adaptive_progress_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {filepath}")

    # Print summary statistics
    print("\n=== OPTIMIZATION ADAPTIVE EVALUATIONS SUMMARY ===")
    print(f"Seed points: {num_seed_points}")
    print(f"Total adaptive evaluations: {len(adaptive_objectives)}")
    print(f"Total iterations: {num_seed_points + len(adaptive_objectives)}")

    if len(adaptive_objectives) > 0:
        extreme_penalty_threshold = 1e5  # Same threshold as used above
        finite_adaptives = np.array(
            [
                obj
                for obj in adaptive_objectives
                if np.isfinite(obj) and obj < extreme_penalty_threshold
            ]
        )
        failed_adaptives = len(
            [
                obj
                for obj in adaptive_objectives
                if not np.isfinite(obj) or obj >= extreme_penalty_threshold
            ]
        )

        print(f"Successful adaptive evaluations: {len(finite_adaptives)}")
        print(f"Failed adaptive evaluations: {failed_adaptives}")

        if finite_adaptives.size > 0:
            print(f"Best seed objective: {best_seed_objective:.3f}")
            print(f"Final best objective: {best_so_far[-1]:.3f}")
            print(f"Best adaptive objective found: {np.min(finite_adaptives):.3f}")
            print(
                f"Mean successful adaptive objective: {np.mean(finite_adaptives):.3f}"
            )
            print(
                f"Std dev successful adaptive objective: {np.std(finite_adaptives):.3f}"
            )

            # Calculate improvement from best seed to final best
            total_improvement = best_seed_objective - best_so_far[-1]
            total_improvement_pct = (total_improvement / best_seed_objective) * 100
            print(f"Total improvement: {total_improvement:.3f}")
            print(f"Total improvement %: {total_improvement_pct:.1f}%")

            # Calculate improvement just from adaptive phase
            if len(finite_adaptives) > 0:
                first_adaptive = (
                    finite_adaptives[0]
                    if finite_adaptives[0] < 1e6
                    else best_seed_objective
                )
                adaptive_improvement = first_adaptive - best_so_far[-1]
                if first_adaptive > 0:
                    adaptive_improvement_pct = (
                        adaptive_improvement / first_adaptive
                    ) * 100
                    print(f"Adaptive phase improvement: {adaptive_improvement:.3f}")
                    print(
                        f"Adaptive phase improvement %: {adaptive_improvement_pct:.1f}%"
                    )

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot optimization progress from CSV results - supports both deterministic and stochastic"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to results CSV file (optional)",
    )
    parser.add_argument(
        "--seed", type=int, default=25, help="Number of seed points (default: 25)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use CSV without any filtration (plot raw best-so-far)",
    )
    args = parser.parse_args()

    # Priority: 1) CSV_FILE_PATH from config, 2) --file argument
    csv_file_to_use = None

    # First, check if CSV_FILE_PATH is configured
    if CSV_FILE_PATH and CSV_FILE_PATH.strip():
        csv_file_to_use = CSV_FILE_PATH.strip()
        print(f"Using configured CSV file: {csv_file_to_use}")
    # Second, check command-line argument
    elif args.file is not None:
        csv_file_to_use = args.file
        print(f"Using command-line file: {csv_file_to_use}")
    # If no file specified, show error
    else:
        print("ERROR: No CSV file specified!")
        print("Please either:")
        print("1. Set CSV_FILE_PATH in the script configuration, or")
        print("2. Use --file <path> argument")
        exit(1)

    # Verify file exists
    if csv_file_to_use and os.path.exists(csv_file_to_use):
        plot_optimization_results(
            csv_file_to_use, num_seed_points=args.seed, allow_filter=not args.raw
        )
    else:
        if csv_file_to_use:
            print(f"Error: File not found: {csv_file_to_use}")
        else:
            print("No results file found!")
        print("Options:")
        print("1. Set CSV_FILE_PATH in the script configuration")
        print("2. Use --file <path> argument")
