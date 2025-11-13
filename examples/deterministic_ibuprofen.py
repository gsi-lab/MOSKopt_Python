"""
Deterministic Optimization Example using MOSKopt with AVEVA Process Simulation.

This example demonstrates deterministic optimization using MOSKopt with
advanced acquisition functions and Particle Swarm Optimization. It optimizes
a chemical process using AVEVA Process Simulation integration, getting
real simulation values from the simulation file.

The example uses:
- **FEI (Feasible Expected Improvement)** acquisition function
- **Particle Swarm Optimization** for infill optimization
- **AVEVA Process Simulation** for process evaluation
- **Gaussian Process Regression** surrogate models
- **Constraint handling** with penalty methods

Custom configuration:
>>> result = run_deterministic_example(
...     max_iterations=75,        # Custom iterations
...     num_seed_points=25,       # Custom seed points
...     num_repetitions=1,        # Deterministic (1 repetition)
...     swarm_size=40,            # Custom PSO swarm size
...     max_iter_pso=40,          # Custom PSO iterations
... )

Notes
This example requires AVEVA Process Simulation and the AVEVA Python interface.
The example connects to the "IbuprofenProcessSimulation" simulation for simulated process evaluation.
"""

import os
import warnings
from datetime import datetime

import numpy as np

from core import AVEVASimulator, StochasticOptimizer

# Suppress scikit-learn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.gaussian_process"
)


def run_deterministic_example(
    max_iterations=75,
    num_seed_points=25,
    num_repetitions=1,
    swarm_size=40,
    max_iter_pso=40,
):
    """
    Run deterministic optimization example with MOSKopt and AVEVA Process Simulation.

    This function demonstrates deterministic optimization using the FEI
    acquisition function and Particle Swarm Optimization. It optimizes
    a chemical process by connecting to AVEVA Process Simulation and
    getting real simulation values from the "IbuprofenProcessSimulation" simulation.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of adaptive iterations. Default: 75
    num_seed_points : int, optional
        Number of initial design points. Default: 25
    swarm_size : int, optional
        PSO swarm size for acquisition function optimization. Default: 10
    max_iter_pso : int, optional
        Maximum PSO iterations for acquisition function optimization. Default: 20
    num_repetitions : int, optional
        Number of Monte Carlo repetitions for uncertainty handling. Default: 1 (deterministic)

    Notes
    -----
    Random seed is fixed to 42 internally for reproducible results.

    Returns
    -------
    dict or OptimizationResult
        Results of the deterministic optimization containing:

        - x : ndarray
            Optimal decision variables [V2, T2, RatioMEKIbap, V1, T1, RatioDecIbap]
        - fun : float
            Optimal objective value (LCO from AVEVA simulation)
        - x_history : ndarray
            History of all evaluated points
        - f_history : ndarray
            History of all objective values
        - g_history : ndarray
            History of all constraint values
        - iteration : int
            Number of iterations completed
        - converged : bool
            Whether optimization converged
        - metadata : dict
            Additional optimization metadata
    """

    def extract_result_data(result, key, default=None):
        """
        Helper function to extract data from result object whether it's a dictionary or object.

        Parameters
        ----------
        result : dict or object
            The result object from optimizer.optimize()
        key : str
            The key/attribute to extract
        default : any, optional
            Default value if key/attribute not found

        Returns
        -------
        any
            The extracted value or default
        """
        if isinstance(result, dict):
            # Handle dictionary results (from compiled core)
            return result.get(key, result.get(f"best_{key}", default))
        else:
            # Handle object results (if available)
            return getattr(result, key, default)

    # ============================================================================
    # OPTIMIZATION CONFIGURATION
    # ============================================================================

    # Define optimization parameters for deterministic optimization
    options = {
        # Optimization control parameters
        "MaxIterations": max_iterations,
        "NumSeedPoints": num_seed_points,
        "NumRepetitions": num_repetitions,
        "InfillCriterion": "FEI",
        "InfillSolver": "particleswarm",
        "Verbose": False,
        "SwarmSize": swarm_size,
        "MaxIterPSO": max_iter_pso,
        "MaxDuplicateAttempts": 5,
        # Alternative infill criteria (users can change these):
        # "InfillCriterion": "cAEI",              # Constrained Augmented Expected Improvement
        # "InfillCriterion": "AEI",               # Augmented Expected Improvement
        # "InfillCriterion": "EI",                # Expected Improvement
        # "InfillCriterion": "UCB",               # Upper Confidence Bound
        # Alternative optimization solvers (users can change these):
        # "InfillSolver": "lbfgs",                # L-BFGS local optimization
        # "InfillSolver": "random",               # Random sampling
        # AVEVA Process Simulation configuration
        "sim_name": "IbuprofenProcessSimulation",
        "snapshot_name": "Pro 1",
        # Independent variables (decision variables)
        "ind_vars": [
            ("R2.V", "m3"),
            ("R2.T2", "K"),
            ("RatioMEKIbap", ""),
            ("R1.V", "m3"),
            ("R1.T2", "K"),
            ("RatioDecIbap", ""),
        ],
        # Uncertainty variables (fixed for deterministic optimization)
        "unc_vars": [("IBAP.RawMatl.PricePerMass", "¤/t")],
        # Dependent variables (outputs from simulation)
        "dep_vars": [
            ("LCO", "¤/t"),
            ("R1.tau", "min"),
            ("R2.tau", "min"),
            ("Yield", ""),
        ],
        "NumCoupledConstraints": 3,
        "CoupledConstraintTolerances": [1e-3] * 3,
        "constraint_names": ["R1.tau", "R2.tau", "Yield"],
        # Constraint configuration: tau1 >= 15, tau2 <= 120, yield >= 0.6
        "constraint_config": [">=", "<=", ">="],
        # Discrete variables (reactor volumes)
        "discrete_vars": {
            0: np.array(
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
            ),
            3: np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        },
        # For deterministic case, IBAP price is fixed at 3000 (no uncertainty)
        "uncertainty_distributions": {
            "IBAP.RawMatl.PricePerMass": {
                "type": "constant",
                "value": 3000.0,
            }
        },
        # Monte Carlo Configuration:
        # - Deterministic optimization: num_repetitions = 1 (no uncertainty, same result every time)
    }

    # Decision variable bounds for the optimization problem
    # These represent the ranges for reactor design parameters
    bounds = [
        (10, 20),  # V2: Volume of reactor 2 (m³) - discrete values allowed
        (358, 408),  # T2: Temperature of reactor 2 (K) - continuous
        (0.1, 0.3),  # RatioMEKIbap: MEK to IBAP ratio - continuous
        (1.5, 5),  # V1: Volume of reactor 1 (m³) - discrete values allowed
        (323, 363),  # T1: Temperature of reactor 1 (K) - continuous
        (0.1, 0.3),  # RatioDecIbap: Decane to IBAP ratio - continuous
    ]

    # Constraint limits: tau1 >= 15, tau2 <= 120, yield >= 0.6
    clims = np.array([15, 120, 0.6])

    # ============================================================================
    # SETUP AND INITIALIZATION
    # ============================================================================

    # Create results directory for plots and data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "optimization_results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    simulator = AVEVASimulator(clims, options)

    # ============================================================================
    # OPTIMIZATION EXECUTION
    # ============================================================================

    # Create optimizer instance with AVEVA simulator
    # print("\nInitializing MOSKopt optimizer...")
    optimizer = StochasticOptimizer(
        bounds=bounds,  # Variable bounds
        options=options,  # Optimization configuration
        simulator=simulator,  # AVEVA simulator for Monte Carlo simulations
    )

    # Run the optimization
    # print("\nStarting deterministic optimization...")
    # print("Using FEI acquisition function with Particle Swarm Optimization")
    result = optimizer.optimize()

    # ============================================================================
    # RESULTS ANALYSIS AND OUTPUT
    # ============================================================================

    # Print optimization results summary
    # print("\n" + "=" * 50)
    # print("OPTIMIZATION RESULTS")
    # print("=" * 50)
    # print(f"Best objective (LCO): {result.fun:.2f} ¤/tonne")
    # print(f"Iterations completed: {result.iteration}")
    # print(f"Converged: {result.converged}")

    # Print optimal decision variables with descriptive names
    var_names = ["V2", "T2", "RatioMEKIbap", "V1", "T1", "RatioDecIbap"]

    # Extract result data using helper function
    best_x = extract_result_data(result, "x")
    x_history = extract_result_data(result, "x_history", [])
    f_history = extract_result_data(result, "f_history", [])
    g_history = extract_result_data(result, "g_history", [])
    iteration = extract_result_data(result, "iteration", 0)
    converged = extract_result_data(result, "converged", False)

    # print("\nOptimal decision variables:")
    # for i, (name, value, unit) in enumerate(zip(var_names, best_x, units)):
    #    if unit:
    #        #print(f"  {name}: {value:.6f} {unit}")
    #    else:
    #       #print(f"  {name}: {value:.6f}")

    # ============================================================================
    # SAVE RESULTS TO CSV FILES
    # ============================================================================

    print("\nSaving results to CSV files...")

    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save optimization history (all iterations)
    history_csv_path = os.path.join(
        results_dir, f"deterministic_adaptive_progress_csv_{timestamp}.csv"
    )
    try:
        import pandas as pd

        # Prepare data for CSV export - ONLY include successful evaluations
        history_data = []
        successful_count = 0
        for i in range(len(x_history)):
            # Skip failed simulations (penalty values >= 1e6)
            objective_val = f_history[i] if i < len(f_history) else np.nan
            if not np.isnan(objective_val) and objective_val < 1e6:
                successful_count += 1
                row = {
                    "Iteration": successful_count,  # Use successful iteration count
                    "LCO_Objective": objective_val,
                    "Converged": converged,
                }

                # Add decision variables
                if i < len(x_history):
                    for j, var_name in enumerate(var_names):
                        row[f"{var_name}"] = (
                            x_history[i][j] if j < len(x_history[i]) else np.nan
                        )

                # Add constraint values
                if i < len(g_history):
                    constraint_names = options.get(
                        "constraint_names",
                        [f"Constraint_{j + 1}" for j in range(len(g_history[i]))],
                    )
                    for j, constraint_name in enumerate(constraint_names):
                        row[f"{constraint_name}_violation"] = (
                            g_history[i][j] if j < len(g_history[i]) else np.nan
                        )

                history_data.append(row)

        # Create DataFrame and save to CSV
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_csv_path, index=False)
        print(
            f"✓ Saved optimization history: {successful_count} successful evaluations to: {history_csv_path}"
        )
        # print(f"✓ Saved optimization history to: {history_csv_path}")

    except Exception as e:
        print(f"⚠ Warning: Could not save history CSV: {e}")
        pass

    # 3. Save summary statistics
    summary_csv_path = os.path.join(
        results_dir, f"deterministic_summary_{timestamp}.csv"
    )
    try:
        # Calculate summary statistics
        if len(f_history) > 0:
            summary_data = {
                "Metric": [
                    "Best_LCO",
                    "Worst_LCO",
                    "Mean_LCO",
                    "Std_LCO",
                    "Total_Iterations",
                    "Converged",
                    "Best_Iteration",
                    "Optimization_Type",
                    "Acquisition_Function",
                    "Solver",
                ],
                "Value": [
                    min(f_history),
                    max(f_history),
                    np.mean(f_history),
                    np.std(f_history),
                    iteration,
                    converged,
                    np.argmin(f_history) + 1 if len(f_history) > 0 else 0,
                    "Deterministic",
                    options["InfillCriterion"],
                    options["InfillSolver"],
                ],
                "Unit": [
                    "¤/tonne",
                    "¤/tonne",
                    "¤/tonne",
                    "¤/tonne",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                "Description": [
                    "Best LCO found during optimization",
                    "Worst LCO found during optimization",
                    "Mean LCO across all iterations",
                    "Standard deviation of LCO across iterations",
                    "Total optimization iterations completed",
                    "Whether optimization converged",
                    "Iteration with best LCO",
                    "Type of optimization performed",
                    "Acquisition function used",
                    "Optimization solver used",
                ],
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_csv_path, index=False)
            # print(f"✓ Saved summary statistics to: {summary_csv_path}")
        else:
            # print("⚠ No objective history to create summary statistics")
            pass

    except Exception as e:
        print(f"⚠ Warning: Could not save summary CSV: {e}")
        pass

    return result


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # print("=" * 80)
    # print("MOSKopt Deterministic Example - FEI + Particle Swarm Optimization")
    # print("=" * 80)

    # Run the deterministic example with default configuration
    # print("\nRunning Deterministic Optimization (FEI + PSO)")
    # print("-" * 50)
    result = run_deterministic_example()

    # Summary of results
    # print("\nOptimization Results Summary:")
    # print(f"Best LCO: {best_fun:.2f} ¤/tonne")
    # print(f"Iterations completed: {iteration}")
    # print(f"Converged: {converged}")

    # Information about alternative configurations
    # print("\nConfiguration Options:")
    # print("   - Infill criteria: 'FEI' (current), 'cAEI', 'AEI', 'EI', 'UCB'")
    # print("   - Optimization solvers: 'particleswarm' (current), 'lbfgs', 'random'")
    # print("   - To use alternatives: Change the 'InfillCriterion' and 'InfillSolver' in options")
    # print("   - PSO parameters are optimized for speed while maintaining quality")

    # print("\nExample completed successfully!")
