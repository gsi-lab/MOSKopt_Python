"""
Deterministic Optimization Example using MOSKopt with AVEVA Process Simulation.

This example demonstrates deterministic optimization using MOSKopt with
advanced acquisition functions and Particle Swarm Optimization. It optimizes
ibuprofen continuous manufacturing process using AVEVA Process Simulation integration, getting
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
The example connects to the "IbuprofenProcessSimulation" simulation for simulated process.
"""

import os
from datetime import datetime

import numpy as np

from core import AVEVASimulator, StochasticOptimizer


def run_deterministic_example(
    max_iterations=75,
    num_seed_points=25,
    num_repetitions=1,
    swarm_size=30,
    max_iter_pso=20,
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
        Maximum number of adaptive iterations.
    num_seed_points : int, optional
        Number of initial design points.
    swarm_size : int, optional
        PSO swarm size for acquisition function optimization.
    max_iter_pso : int, optional
        Maximum PSO iterations for acquisition function optimization.
    num_repetitions : int, optional
        Number of Monte Carlo repetitions for uncertainty handling.

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
        "MaxDuplicateAttempts": 3,
        # Alternative infill criteria (users can change these):
        # "InfillCriterion": "cAEI",        # Alternative infill criteria (users can change these):
        # "InfillCriterion": "mcFEI",             # multiple constrained Feasibility Enhanced Improvement
        # "InfillCriterion": "EI",                # Expected Improvement
        # "InfillCriterion": "AEI",               # Augmented Expected Improvement
        # "InfillCriterion": "cAEI",              # Constrained Augmented Expected Improvement
        # "InfillCriterion": "FEI",               # Feasibility Enhanced Constrained Improvement tion",
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
        "expected_constraints": 3,
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

    # Create results directory for plots and data (in project root optimization_results folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        script_dir
    )  # Go up one level from examples to project root
    results_dir = os.path.join(project_root, "optimization_results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Initialize AVEVA simulator for deterministic optimization
    # print("\nInitializing AVEVA Process Simulation connection...")
    simulator = AVEVASimulator(clims, options)
    # print("✓ AVEVA simulator initialized successfully")
    # print(f"  Simulation file: {options['sim_name']}")
    # print(f"  Snapshot: {options['snapshot_name']}")

    # Test AVEVA connection before proceeding
    try:
        # Try to get a simple test simulation to verify connection
        test_x = np.array([[15.0, 383.0, 0.2, 3.0, 343.0, 0.2]])  # Test point
        test_f, test_g, test_status, test_f_std, test_g_std = simulator.simulate(test_x)

        # Check if we got valid results (not fallback simulation)
        if not test_status[0] or np.isnan(test_f[0]) or test_f[0] >= 1e6:
            print("\n" + "=" * 60)
            print("❌ AVEVA SIMULATION NOT AVAILABLE")
            print("=" * 60)
            print("The optimization could not connect to AVEVA Process Simulation.")
            print("Please ensure:")
            print("  1. AVEVA Process Simulation is running")
            print("  2. The simulation file is open")
            print("  3. AVEVA Python interface is properly configured")
            print("\nThe example requires AVEVA simulation to run successfully.")
            print("=" * 60)
            return None

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ AVEVA SIMULATION CONNECTION FAILED")
        print("=" * 60)
        print(f"Error connecting to AVEVA: {e}")
        print("Please ensure:")
        print("  1. AVEVA Process Simulation is running")
        print("  2. The simulation file is open")
        print("  3. AVEVA Python interface is properly configured")
        print("\nThe example requires AVEVA simulation to run successfully.")
        print("=" * 60)
        return None

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

    # Constraint verification is handled internally by the simulator
    constraint_names = ["tau1 >= 15 min", "tau2 <= 120 min", "yield >= 0.6"]
    # Final constraint values are available in the optimization results if needed
    # ============================================================================
    # SAVE RESULTS TO CSV FILES
    # ============================================================================

    # print("\nSaving results to CSV files...")

    # Use the same results directory as the main optimization (project root optimization_results folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        script_dir
    )  # Go up one level from examples to project root
    results_dir = os.path.join(project_root, "optimization_results")
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

    except Exception:
        # print(f"⚠ Warning: Could not save history CSV: {e}")
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

    except Exception:
        # print(f"⚠ Warning: Could not save summary CSV: {e}")
        pass

    # 4. Save failure analysis
    failure_csv_path = os.path.join(
        results_dir, f"deterministic_failure_analysis_{timestamp}.csv"
    )
    try:
        # Calculate failure statistics from optimization results
        total_evaluations = len(x_history)
        # Count failed simulations from f_history (penalty values >= 1e6)
        failed_evaluations = sum(1 for f in f_history if f >= 1e6)
        failure_rate = (
            failed_evaluations / total_evaluations if total_evaluations > 0 else 0
        )

        failure_data = {
            "Metric": [
                "Total_Evaluations",
                "Failed_Evaluations",
                "Failure_Rate",
                "Successful_Evaluations",
                "Success_Rate",
            ],
            "Value": [
                total_evaluations,
                failed_evaluations,
                f"{failure_rate:.2%}",
                total_evaluations - failed_evaluations,
                f"{(1 - failure_rate):.2%}",
            ],
            "Description": [
                "Total number of function evaluations",
                "Number of failed simulations",
                "Percentage of failed evaluations",
                "Number of successful simulations",
                "Percentage of successful evaluations",
            ],
        }

        failure_df = pd.DataFrame(failure_data)
        failure_df.to_csv(failure_csv_path, index=False)
        # print(f"✓ Saved failure analysis to: {failure_csv_path}")

    except Exception:
        # print(f"⚠ Warning: Could not save failure analysis CSV: {e}")
        pass

    # print(f"\n📁 All CSV files saved to: {results_dir}")
    # print("   - deterministic_adaptive_progress_csv_[timestamp].csv: Complete optimization trajectory")
    # print("   - deterministic_final_results_[timestamp].csv: Optimal solution details")
    # print("   - deterministic_summary_[timestamp].csv: Key performance metrics")
    # print("   - deterministic_failure_analysis_[timestamp].csv: Failure analysis")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    # Calculate and display failure statistics from optimization results
    total_evaluations = len(x_history)
    # Count failed simulations from f_history (penalty values >= 1e6)
    failed_evaluations = sum(1 for f in f_history if f >= 1e6)
    failure_rate = (
        failed_evaluations / total_evaluations if total_evaluations > 0 else 0
    )

    # print("\n" + "=" * 60)
    # print("DETERMINISTIC OPTIMIZATION COMPLETED SUCCESSFULLY!")
    # print("=" * 60)
    # print(f"Best LCO achieved: {best_fun:.2f} ¤/tonne")
    # print(f"Iterations completed: {iteration}")
    # print(f"Optimization converged: {converged}")
    # print(f"Results saved to: {results_dir}")
    # print(f"\nFailure Analysis:")
    # print(f"  Total evaluations: {total_evaluations}")
    # print(f"  Failed simulations: {failed_evaluations}")
    # print(f"  Failure rate: {failure_rate:.2%}")
    # print(f"  Success rate: {(1 - failure_rate):.2%}")
    # print("\nNext steps:")
    # print("   - Analyze CSV files for detailed results")
    # print("   - Use plot_deterministic_from_csv.py for visualization")
    # print("   - Modify parameters in options for different configurations")
    # print("   - Try different acquisition functions or solvers")

    return result


if __name__ == "__main__":
    # Set random seed for reproducibility
    # This ensures consistent results across different runs
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

    print("\nExample completed successfully!")
