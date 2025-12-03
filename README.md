# MOSKopt_Python

Advanced simulation-based optimization framework with AVEVA Process Simulation integration, featuring enhanced acquisition functions and robust failure handling.

## Project Purpose and Context

MOSKopt_Python is a Python implementation of the MOSKopt (Simulation-Based Stochastic Kriging Optimization framework), designed for chemical process optimization under uncertainty. This implementation extends the original MATLAB version developed by Resul Al. (https://github.com/gsi-lab/MOSKopt) with enhanced features and performance optimizations. 

### Key Differences from MATLAB Version

- **Enhanced failure handling** with smart consecutive failure limits
- **Intelligent restart strategies** using successful simulation history (snapshot)
- **Performance optimizations** (reduced model refitting, batch predictions)
- **AVEVA Python interface integration** with unified base classes
- **Modern Python features** (type hints, dataclasses, comprehensive error handling)

## Quick Start

### 1. Download the Repository
```bash
# Download as ZIP from GitHub (green "Code" button → "Download ZIP")
# Extract the ZIP file to your desired location
cd MOSKopt_Python-main
```

### 2. Install the Package
```bash
pip install -e .
```

### 3. Run Examples
```bash
python examples/deterministic_ibuprofen.py
python examples/stochastic_ibuprofen.py
```

## Prerequisites
- **Python 3.10 or Python 3.11 or Python 3.12**
- **AVEVA Process Simulation** 
- **AVEVA Python Interface** (`simcentralconnect`)

## AVEVA Custom Libraries Setup for Ibuprofen Example
For the AVEVA Process Simulation examples to run correctly, you need to place the provided simulation library files into your My Thermo Data directory on your computer:
Pharma.BASE.cmp
Pharma.bnk
Pharma.lb1
Pharma.lb2

## Folder Structure

```
MOSKopt_Python/
├── core/                         
│   ├── __init__.py                   # Package initialization
│   ├── combined_core.cpython-310.pyc # Core implementation (compiled with Python 3.10)
│   ├── combined_core.cpython-311.pyc # Core implementation (compiled with Python 3.11)
│   └── combined_core.cpython-312.pyc # Core implementation (compiled with Python 3.12)
├── examples/                     
│   ├── deterministic_ibuprofen.py    # Deterministic optimization example
│   └── stochastic_ibuprofen.py      # Stochastic optimization example
├── simulation/ 
│   ├── IbuprofenProcessSimulation
│   ├── Pharma/BASE/
│   ├── Pharma.BASE.cmp
│   ├── Pharma.bnk
│   ├── Pharma.lb1
│   ├── Pharma.lb2
├── plot_optimization_results.py  # Results plotting tool
├── setup.py                      # Package setup
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## Acknowledgement

This Python implementation was supported by European Marie Skłodowska-Curie network MiEl. The MiEl project received funding by the European Union under the Grant Agreement no. 101073003. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

**Note**: Core implementation is compiled to `.pyc` files for intellectual property protection. Examples and documentation are provided for easy usage. 
