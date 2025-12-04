"""
Stochastic Optimization Example using MOSKopt with AVEVA Process Simulation.

This example demonstrates stochastic optimization using MOSKopt with
Monte Carlo uncertainty quantification and AVEVA Process Simulation integration.
It optimizes ibuprofen continuous manufacturing process while handling uncertainty in raw material price
and simulation failures.

The example uses:
- mcFEI (Monte Carlo Feasible Expected Improvement) acquisition function
- Particle Swarm Optimization for infill optimization
- Monte Carlo sampling for uncertainty quantification
- AVEVA Process Simulation for process evaluation
- Robust failure handling and simulation restart strategies

Custom configuration:
>>> result = run_aveva_example(
...     max_iterations=75,        # Custom iterations
...     num_seed_points=25,       # Custom seed points
...     num_repetitions=100,      # Monte Carlo samples
...     swarm_size=40,            # Custom PSO swarm size
...     max_iter_pso=40,          # Custom PSO iterations
... )

Notes
-----
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

# Note: This example uses the core AVEVASimulator from moskopt.core._simulator
# Users can modify the simulation configuration below for their own project


def run_aveva_example(
    max_iterations=75,  # Full optimization iterations
    num_seed_points=25,  # Proper initial design for stochastic
    num_repetitions=100,  # Good balance of MC samples for uncertainty
    swarm_size=40,  # Proper PSO size for stochastic optimization
    max_iter_pso=40,  # Proper PSO iterations for quality
):
    """
    Run stochastic optimization example with MOSKopt and AVEVA Process Simulation.

    This function demonstrates stochastic optimization using the mcFEI
    acquisition function and Particle Swarm Optimization. It optimizes
    a chemical process by connecting to AVEVA Process Simulation and
    handling uncertainty in raw material prices and simulation failures.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of adaptive iterations.
    num_seed_points : int, optional
        Number of initial design points.
    num_repetitions : int, optional
        Number of Monte Carlo samples for uncertainty.
    swarm_size : int, optional
        PSO swarm size for acquisition function optimization.
    max_iter_pso : int, optional
        Maximum PSO iterations for acquisition function optimization.

    Returns
    -------
    dict or OptimizationResult
        Results of the stochastic optimization containing:

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

    # Define optimization parameters for stochastic optimization
    options = {
        # Optimization control parameters
        "MaxIterations": max_iterations,  # Number of adaptive iterations
        "NumSeedPoints": num_seed_points,  # Number of initial design points
        "NumRepetitions": num_repetitions,  # Monte Carlo samples for uncertainty
        "InfillCriterion": "mcFEI",  # multiple constrained FEI (fixed)
        "InfillSolver": "particleswarm",  # Particle Swarm Optimization (fixed)
        "Verbose": False,  # Enable progress output
        "SwarmSize": swarm_size,  # PSO swarm size (optimized default)
        "MaxIterPSO": max_iter_pso,  # Maximum PSO iterations (optimized default)
        "MaxDuplicateAttempts": 3,  # Maximum attempts to find unique points
        # Optimization direction
        "Maximize": False,  # Set to True for maximization, False for minimization (default)
        # Constraint settings
        "NumCoupledConstraints": 3,  # Number of constraints (can be set to 0 if no constraints)
        "CoupledConstraintTolerances": [1e-3]
        * 3,  # Constraint tolerances (can be empty list [] if no constraints)
        # Alternative infill criteria (users can change these):
        # "InfillCriterion": "mcFEI",             # multiple constrained Feasibility Enhanced Improvement (for constrained problems)
        # "InfillCriterion": "EI",                # Expected Improvement (for unconstrained problems)
        # "InfillCriterion": "AEI",               # Augmented Expected Improvement (for stochastic problems)
        # "InfillCriterion": "cAEI",              # Constrained Augmented Expected Improvement (for constrained stochastic problems)
        # "InfillCriterion": "FEI",               # Feasibility Enhanced Constrained Improvement (for highly constrained problems)
        # Note: For unconstrained problems, set NumCoupledConstraints=0 and use EI or AEI infill criteria
        # AVEVA Process Simulation configuration
        "sim_name": "IbuprofenProcessSimulation",  # AVEVA simulation file name
        "snapshot_name": "Pro 1",  # AVEVA snapshot name (single snapshot mode)
        # Dual snapshot mode (optional - for different operating conditions):
        # "snapshot_name_1": "Pro 1",     # Snapshot 1 (e.g., for uncertain_var > threshold)
        # "snapshot_name_2": "Pro 2",     # Snapshot 2 (e.g., for uncertain_var <= threshold)
        # "snapshot_threshold": 150.0,    # Threshold value for snapshot selection
        # Independent variables (decision variables)
        "ind_vars": [
            ("R2.V", "m3"),  # V2: Volume of reactor 2
            ("R2.T2", "K"),  # T2: Temperature of reactor 2
            ("RatioMEKIbap", ""),  # MEK to IBAP ratio
            ("R1.V", "m3"),  # V1: Volume of reactor 1
            ("R1.T2", "K"),  # T1: Temperature of reactor 1
            ("RatioDecIbap", ""),  # Decane to IBAP ratio
        ],
        # Uncertainty variables (Monte Carlo sampled)
        "unc_vars": [
            ("IBAP.RawMatl.PricePerMass", "¤/t")  # IBAP raw material price
        ],
        # Dependent variables (outputs from simulation)
        "dep_vars": [
            ("LCO", "¤/t"),  # Levelized Cost of Operation
            ("R1.tau", "min"),  # Residence time in reactor 1
            ("R2.tau", "min"),  # Residence time in reactor 2
            ("Yield", ""),  # Product yield
        ],
        "constraint_names": ["R1.tau", "R2.tau", "Yield"],
        # Constraint configuration: tau1 >= 15, tau2 <= 120, yield >= 0.6
        "constraint_config": [">=", "<=", ">="],
        # Discrete variables (reactor volumes)
        "discrete_vars": {
            0: np.array(
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
            ),  # V2
            3: np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),  # V1
        },
        # Uncertainty distributions for Monte Carlo sampling
        # Match standalone code exactly: fit lognormal to bounds 2700-3900
        "uncertainty_distributions": {
            "IBAP.RawMatl.PricePerMass": {
                "type": "lognormal",
                "median": 3000.0,
                "lower_bound": 2700.0,  # 0.9 * 3000
                "upper_bound": 3900.0,  # 1.3 * 3000
                "lower_percentile": 5,
                "upper_percentile": 95,
            }
        },
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

    # Initialize AVEVA simulator for stochastic optimization
    # print("\nInitializing AVEVA Process Simulation connection...")
    simulator = AVEVASimulator(clims, options)
    # print("✓ AVEVA simulator initialized successfully")
    # print(f"  Simulation file: {options['sim_name']}")
    # print(f"  Snapshot: {options['snapshot_name']}")

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
    # print("\nStarting stochastic optimization...")
    # print("Using mcFEI acquisition function with Particle Swarm Optimization")
    result = optimizer.optimize()

    # ============================================================================
    # RESULTS ANALYSIS AND OUTPUT
    # ============================================================================

    # Print optimization results summary
    # print("\n" + "=" * 50)
    # print("OPTIMIZATION RESULTS")
    # print("=" * 50)
    # print(f"Iterations completed: {iteration}")
    # print(f"Converged: {converged}")

    # Print optimal decision variables with descriptive names
    var_names = ["V2", "T2", "RatioMEKIbap", "V1", "T1", "RatioDecIbap"]
    constraint_names = ["R1.tau", "R2.tau", "Yield"]  # Match options configuration

    # Extract result data using helper function
    x_history = extract_result_data(result, "x_history", [])
    f_history = extract_result_data(result, "f_history", [])
    g_history = extract_result_data(result, "g_history", [])
    iteration = extract_result_data(result, "iteration", 0)
    converged = extract_result_data(result, "converged", False)
    # best_idx = extract_result_data(result, "best_idx", None)

    # ============================================================================
    # SAVE RESULTS TO CSV FILES
    # ============================================================================

    # print("\nSaving results to CSV files...")

    # Generate timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save optimization history (all iterations)
    history_csv_path = os.path.join(
        results_dir, f"stochastic_adaptive_progress_csv_{timestamp}.csv"
    )
    try:
        import pandas as pd

        # Prepare data for CSV export
        history_data = []
        for i in range(len(x_history)):
            row = {
                "Iteration": i + 1,
                "LCO_Objective": f_history[i] if i < len(f_history) else np.nan,
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
        print(f"✓ Saved adaptive progress to: {history_csv_path}")

    except Exception:
        # print(f"⚠ Warning: Could not save history CSV: {e}")
        pass

    # 3. Save summary statistics
    summary_csv_path = os.path.join(results_dir, f"stochastic_summary_{timestamp}.csv")
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
                    "Monte_Carlo_Samples",
                    "Failure_Handling",
                ],
                "Value": [
                    min(f_history),
                    max(f_history),
                    np.mean(f_history),
                    np.std(f_history),
                    iteration,
                    converged,
                    np.argmin(f_history) + 1 if len(f_history) > 0 else 0,
                    "Stochastic",
                    options["InfillCriterion"],
                    options["InfillSolver"],
                    num_repetitions,
                    "Smart consecutive failure limits + restart strategies",
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
                    "Number of Monte Carlo samples for uncertainty",
                    "Failure handling strategy used",
                ],
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_csv_path, index=False)
            # print(f"✓ Saved summary statistics to: {summary_csv_path}")
        # else:
        # print("⚠ No objective history to create summary statistics")

    except Exception:
        # print(f"⚠ Warning: Could not save summary CSV: {e}")
        pass

    # 4. Save Monte Carlo uncertainty details
    uncertainty_csv_path = os.path.join(
        results_dir, f"monte_carlo_uncertainty_{timestamp}.csv"
    )
    try:
        # Create uncertainty details DataFrame
        uncertainty_data = {
            "Parameter": ["IBAP_Price"],
            "Distribution": ["Lognormal"],
            "Median": [3000.0],
            "Unit": ["¤/tonne"],
            "Samples": [num_repetitions],
            "Min_Value": [simulator.Pu.min()],
            "Max_Value": [simulator.Pu.max()],
            "Description": ["IBAP raw material price uncertainty"],
        }

        uncertainty_df = pd.DataFrame(uncertainty_data)
        uncertainty_df.to_csv(uncertainty_csv_path, index=False)
        # print(f"✓ Saved uncertainty details to: {uncertainty_csv_path}")

    except Exception:
        # print(f"⚠ Warning: Could not save uncertainty CSV: {e}")
        pass

    return result


if __name__ == "__main__":
    np.random.seed(42)
    # print("Set fixed random seed (42) for reproducible results")

    # print("=" * 80)
    # print("MOSKopt Stochastic Example - mcFEI + Particle Swarm Optimization")
    # print("=" * 80)

    # Run the stochastic example with default configuration
    # print("\nRunning Stochastic Optimization (mcFEI + PSO)")
    # print("-" * 50)
    result = run_aveva_example()

    # Summary of results
    # print("\nOptimization Results Summary:")
    # print(f"Iterations completed: {iteration}")
    # print(f"Converged: {converged}")

    print("\nExample completed successfully!")
