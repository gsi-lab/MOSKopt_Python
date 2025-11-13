"""
MOSKopt Core Module

This module contains the core implementation of MOSKopt algorithms.
All core classes are compiled to a single .pyc file for intellectual property protection.

Classes provided:
- InternalStochasticOptimizer: Core optimization engine
- Simulator: Base simulator class  
- AVEVASimulator: AVEVA Process Simulation integration
- SurrogateModels: Kriging and Gaussian Process models
- AcquisitionFunctions: Various infill criteria (FEI, mcFEI, etc.)
"""

import os
import sys
import importlib.util

def load_compiled_core():
    """Load the compiled core module based on Python version."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Detect Python version and choose appropriate .pyc file
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        core_filename = f"combined_core.cpython-{python_version.replace('.', '')}.pyc"
        core_path = os.path.join(current_dir, core_filename)
        
        # If version-specific file doesn't exist, try to find any available .pyc file
        if not os.path.exists(core_path):
            available_files = [f for f in os.listdir(current_dir) if f.startswith("combined_core.cpython-") and f.endswith(".pyc")]
            if available_files:
                core_filename = available_files[0]  # Use the first available one
                core_path = os.path.join(current_dir, core_filename)
                print(f"Warning: Using {core_filename} for Python {python_version}")
            else:
                raise FileNotFoundError(f"No compiled core files found in {current_dir}")
        
        # Load the compiled .pyc file
        spec = importlib.util.spec_from_file_location("combined_core", core_path)
        if spec is None:
            raise ImportError("Could not create spec for compiled core module")
        
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        
        return core_module
        
    except Exception as e:
        print(f"Error: Could not load compiled core module: {e}")
        print("Please ensure the combined_core.cpython-*.pyc file exists in the core directory")
        return None

# Try to load the compiled core module
core_module = load_compiled_core()

if core_module is not None:
    # Import all classes from the compiled module
    try:
        InternalStochasticOptimizer = core_module.InternalStochasticOptimizer
        Simulator = core_module.Simulator
        AVEVASimulator = core_module.AVEVASimulator
        SurrogateModels = core_module.SurrogateModels
        AcquisitionFunctions = core_module.AcquisitionFunctions
        NoisePredictor = core_module.NoisePredictor
        
        # Create alias for backward compatibility
        StochasticOptimizer = InternalStochasticOptimizer
        
        print("✓ Successfully loaded compiled MOSKopt core module!")
        
    except AttributeError as e:
        print(f"Error: Compiled module missing required class: {e}")
        print("Please check that the combined_core.pyc file contains all required classes")
        core_module = None

if core_module is None:
    # Fallback classes if loading failed
    class InternalStochasticOptimizer:
        def __init__(self, *args, **kwargs):
            print("ERROR: Core optimizer module could not be loaded.")
            print("Please ensure the combined_core.cpython-310.pyc file exists and is valid")
            raise ImportError("Core optimizer module not available. Please check the compiled core file.")

    class StochasticOptimizer(InternalStochasticOptimizer):
        pass

    class Simulator:
        def __init__(self, *args, **kwargs):
            print("ERROR: Core simulator module could not be loaded.")
            print("Please ensure the combined_core.cpython-310.pyc file exists and is valid")
            raise ImportError("Core simulator module not available. Please check the compiled core file.")

    class AVEVASimulator(Simulator):
        pass

    class SurrogateModels:
        def __init__(self, *args, **kwargs):
            print("ERROR: Core surrogate models module could not be loaded.")
            print("Please ensure the combined_core.cpython-310.pyc file exists and is valid")
            raise ImportError("Core surrogate models module not available. Please check the compiled core file.")

    class AcquisitionFunctions:
        def __init__(self, *args, **kwargs):
            print("ERROR: Core acquisition functions module could not be loaded.")
            print("Please ensure the combined_core.cpython-310.pyc file exists and is valid")
            raise ImportError("Core acquisition functions module not available. Please check the compiled core file.")

    class NoisePredictor:
        def __init__(self, *args, **kwargs):
            print("ERROR: Core noise predictor module could not be loaded.")
            print("Please ensure the combined_core.cpython-310.pyc file exists and is valid")
            raise ImportError("Core noise predictor module not available. Please check the compiled core file.")

# Always ensure these classes are available
__all__ = [
    'InternalStochasticOptimizer',
    'StochasticOptimizer',  # Alias for examples
    'Simulator', 
    'AVEVASimulator',
    'SurrogateModels', 
    'AcquisitionFunctions',
    'NoisePredictor'
]
