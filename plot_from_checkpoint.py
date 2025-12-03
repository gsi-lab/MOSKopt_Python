import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import argparse
import os
import sys
from datetime import datetime

# ============================================================================
# CONFIGURATION: Just paste your checkpoint CSV file path here and run!
# ============================================================================
CSV_FILE_PATH = r"C:\Users\tusas\OneDrive - Danmarks Tekniske Universitet\Skrivebord\CursorCodes\Paper1codes\MOSKopt_Python\optimization_results\checkpoint_20251130_064358_iter_70.csv"  # <-- Paste your checkpoint CSV file path between the quotes

# Example: CSV_FILE_PATH = r"C:\Users\...\optimization_results\checkpoint_20251203_000642_iter_70.csv"
# ============================================================================


def plot_from_checkpoint(csv_file, save_dir="optimization_results", num_seed_points=15):
    """
    Plot optimization progress from checkpoint CSV files.

    Checkpoint files contain ALL evaluation attempts (successful and failed) with Status column.
    This function filters and processes the data to show clean optimization progress.

    Parameters
    ----------
    csv_file : str
        Path to checkpoint CSV file
    save_dir : str
        Directory to save the plot
    num_seed_points : int
        Number of seed points in the optimization
    """

    # Load CSV results
    df = pd.read_csv(csv_file)
    original_size = len(df)

    print(f"Loaded checkpoint data: {len(df)} total evaluation attempts")
    print(f"Columns: {df.columns.tolist()}")

    # Detect objective column name
    objective_col = None
    possible_obj_names = ["LCO_Objective", "Objective", "obj", "LCO"]
    for col_name in possible_obj_names:
        if col_name in df.columns:
            objective_col = col_name
            break

    if objective_col is None:
        print(
            f"Error: No objective column found. Available columns: {df.columns.tolist()}"
        )
        return None

    print(f"Using objective column: {objective_col}")

    # Analyze checkpoint data structure
    if "Status" in df.columns:
        success_count = (df["Status"] == "Success").sum()
        failed_count = (df["Status"] == "Failed").sum()
        print(f"Success evaluations: {success_count}")
        print(f"Failed evaluations: {failed_count}")
        print(f"Success rate: {success_count / len(df) * 100:.1f}%")

        # Filter for successful evaluations only
        df_successful = df[df["Status"] == "Success"].copy().reset_index(drop=True)
        print(f"Using {len(df_successful)} successful evaluations for plotting")
    else:
        print(
            "Warning: No Status column found - assuming all evaluations are successful"
        )
        df_successful = df.copy()

    # Prepare iteration numbers for successful evaluations (use original Iteration column if present)
    if "Iteration" in df_successful.columns:
        try:
            iter_nums = df_successful["Iteration"].astype(int).tolist()
        except Exception:
            iter_nums = list(range(1, len(df_successful) + 1))
    else:
        iter_nums = list(range(1, len(df_successful) + 1))

    # Auto-detect number of seed points if not specified or if mismatch
    if len(df_successful) > 0:
        # Try to detect seed points by looking for patterns in objective values
        objectives = pd.to_numeric(df_successful[objective_col], errors="coerce")

        # For ibuprofen: seed points typically in 6000-7000 range, adaptive improvements below 6000
        # For AICR: different objective ranges

        # Simple heuristic: if we have more data than expected seed points,
        # and there's a clear improvement trend, auto-detect seed boundary
        if len(df_successful) > num_seed_points:
            # Look for the point where consistent improvement starts
            rolling_min = objectives.rolling(window=5, min_periods=1).min()
            improvement_starts = []

            for i in range(num_seed_points, min(len(objectives), num_seed_points + 20)):
                if i < len(rolling_min) - 5:
                    recent_trend = rolling_min[i : i + 5]
                    if len(recent_trend) > 1 and (
                        recent_trend.iloc[-1] < recent_trend.iloc[0]
                    ):
                        improvement_starts.append(i)

            if improvement_starts:
                detected_seed_end = min(improvement_starts)
                print(f"Auto-detected seed phase end at evaluation {detected_seed_end}")
                num_seed_points = detected_seed_end

    print(f"Using {num_seed_points} seed points for analysis")

    # Split into seed and adaptive phases
    seed_data = (
        df_successful.iloc[:num_seed_points]
        if num_seed_points <= len(df_successful)
        else df_successful
    )
    adaptive_data = (
        df_successful.iloc[num_seed_points:].reset_index(drop=True)
        if num_seed_points < len(df_successful)
        else pd.DataFrame()
    )

    print(f"Seed evaluations: {len(seed_data)}")
    print(f"Adaptive evaluations: {len(adaptive_data)}")

    # Check for constraints
    constraint_columns = [
        col for col in df_successful.columns if col.endswith("_violation")
    ]
    if not constraint_columns:
        constraint_columns = [
            col for col in df_successful.columns if col.startswith("Constraint_")
        ]

    # Determine feasibility
    if len(constraint_columns) > 0:
        print(
            f"Found {len(constraint_columns)} constraint columns: {constraint_columns}"
        )
        seed_feasible_mask = (seed_data[constraint_columns].fillna(0) <= 0).all(axis=1)
        feasible_seed_data = seed_data[seed_feasible_mask]

        if len(adaptive_data) > 0:
            adaptive_feasible_mask = (
                adaptive_data[constraint_columns].fillna(0) <= 0
            ).all(axis=1)
            feasible_adaptive_data = adaptive_data[adaptive_feasible_mask]
        else:
            feasible_adaptive_data = pd.DataFrame()
    else:
        print("No constraints detected - treating all points as feasible")
        feasible_seed_data = seed_data
        feasible_adaptive_data = adaptive_data

    # Get best seed objective
    if len(feasible_seed_data) > 0:
        best_seed_objective = feasible_seed_data[objective_col].min()
        print(f"Best feasible seed objective: {best_seed_objective:.3f}")
    else:
        best_seed_objective = seed_data[objective_col].min()
        print(f"Best seed objective (may be infeasible): {best_seed_objective:.3f}")

    # Create improvement tracking for adaptive phase
    # Use actual iteration numbers from `iter_nums` so plotted x-values match the CSV
    improvement_iterations = [iter_nums[num_seed_points - 1]]
    improvement_objectives = [best_seed_objective]
    current_best = best_seed_objective

    # Iterate over successful evaluations after the seed phase using their original iteration numbers
    for pos in range(num_seed_points, len(df_successful)):
        obj_value = df_successful.iloc[pos][objective_col]
        actual_iteration = iter_nums[pos]

        if obj_value < current_best:
            improvement_iterations.append(actual_iteration)
            improvement_objectives.append(obj_value)
            current_best = obj_value
            print(f"Improvement at iteration {actual_iteration}: {obj_value:.3f}")

    # Add final point (last successful evaluation) if not already added
    final_iteration = iter_nums[-1]
    if improvement_iterations[-1] != final_iteration:
        improvement_iterations.append(final_iteration)
        improvement_objectives.append(current_best)

    print(f"Total improvements found: {len(improvement_iterations) - 1}")
    print(f"Final best objective: {improvement_objectives[-1]:.3f}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot improvement progress
    ax.plot(
        improvement_iterations,
        improvement_objectives,
        "bo-",
        linewidth=3,
        markersize=8,
        label=f"Optimization Progress (Final: {improvement_objectives[-1]:.1f})",
    )

    # Add step-wise connection
    ax.step(
        improvement_iterations,
        improvement_objectives,
        "b-",
        where="post",
        alpha=0.7,
        linewidth=2,
    )

    ax.set_xlabel("Iteration Number", fontsize=20)
    ax.set_ylabel("Best Objective Value", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=18)

    # Determine plot title based on data characteristics
    total_attempts = original_size
    successful_attempts = len(df_successful)
    plot_title = f"Optimization Progress from Checkpoint (Attempts: {total_attempts}, Successful: {successful_attempts})"

    ax.set_title(plot_title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=14)

    # Set x-axis limits
    ax.set_xlim(max(0, num_seed_points - 2), max(improvement_iterations) + 5)

    # Add improvement markers
    for i, (iter_num, obj_val) in enumerate(
        zip(improvement_iterations[1:], improvement_objectives[1:])
    ):
        ax.annotate(
            f"{obj_val:.0f}",
            (iter_num, obj_val),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=10,
            alpha=0.8,
        )

    # Add summary annotation
    initial_value = improvement_objectives[0]
    final_value = improvement_objectives[-1]
    total_improvement = initial_value - final_value

    annotation_text = f"Best Seed: {initial_value:.0f}\nFinal Best: {final_value:.0f}\nTotal Improvement: {total_improvement:.0f}\nSuccess Rate: {successful_attempts}/{total_attempts} ({successful_attempts / total_attempts * 100:.1f}%)"
    ax.annotate(
        annotation_text,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_progress_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Checkpoint plot saved to: {filepath}")

    # Print detailed summary
    print("\n=== CHECKPOINT ANALYSIS SUMMARY ===")
    print(f"Total evaluation attempts: {total_attempts}")
    print(f"Successful evaluations: {successful_attempts}")
    print(f"Failed evaluations: {total_attempts - successful_attempts}")
    print(f"Success rate: {successful_attempts / total_attempts * 100:.1f}%")
    print(f"Seed points analyzed: {len(seed_data)}")
    print(f"Adaptive points analyzed: {len(adaptive_data)}")
    print(f"Feasible seed points: {len(feasible_seed_data)}")
    print(f"Feasible adaptive points: {len(feasible_adaptive_data)}")
    print(f"Best initial objective: {initial_value:.3f}")
    print(f"Final best objective: {final_value:.3f}")

    if len(improvement_objectives) > 1:
        improvement_pct = (total_improvement / initial_value) * 100
        print(f"Total improvement: {total_improvement:.3f} ({improvement_pct:.1f}%)")

    plt.close(fig)
    return fig


def find_latest_checkpoint():
    """Find the most recent checkpoint file"""
    current_dir = os.getcwd()
    checkpoint_dir = os.path.join(current_dir, "optimization_results")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".csv"):
            full_path = os.path.join(checkpoint_dir, file)
            checkpoint_files.append(full_path)

    if not checkpoint_files:
        print("No checkpoint files found!")
        return None

    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]

    print(f"Found {len(checkpoint_files)} checkpoint files")
    print(f"Using most recent checkpoint: {os.path.basename(latest_checkpoint)}")
    return latest_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot optimization progress from checkpoint CSV files"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to checkpoint CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=25,
        help="Number of seed points (default: 25, use 15 for AICR)",
    )
    args = parser.parse_args()

    # Priority: 1) CSV_FILE_PATH from config, 2) command-line argument
    csv_file_to_use = None

    # First, check if CSV_FILE_PATH is configured
    if CSV_FILE_PATH and CSV_FILE_PATH.strip():
        csv_file_to_use = CSV_FILE_PATH.strip()
        print(f"Using configured CSV file: {csv_file_to_use}")
    # Second, check command-line argument
    elif args.file is not None:
        csv_file_to_use = args.file
        print(f"Using command-line CSV file: {csv_file_to_use}")
    # If no file specified, show error
    else:
        print("ERROR: No checkpoint CSV file specified!")
        print("Please either:")
        print("1. Set CSV_FILE_PATH in the script configuration, or")
        print("2. Use --file <path> argument")
        sys.exit(1)

    # Verify file exists
    if not csv_file_to_use:
        # This should not happen due to earlier check, but just in case
        sys.exit(1)

    if not os.path.exists(csv_file_to_use):
        print(f"Error: Checkpoint CSV file not found: {csv_file_to_use}")
        sys.exit(1)

    plot_from_checkpoint(csv_file_to_use, num_seed_points=args.seed)
