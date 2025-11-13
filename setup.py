"""
Setup configuration for MOSKopt package.

This file defines the package metadata, dependencies, and structure for
distribution and installation. It uses setuptools to create a proper
Python package that can be installed via pip or other package managers.

The package includes:
- Core optimization algorithms (compiled to .pyc for IP protection)
- Example scripts for deterministic and stochastic optimization
- AVEVA Process Simulation integration
- Comprehensive documentation and user guides

Examples
--------
Install in development mode:
>>> pip install -e .

Install from source:
>>> pip install .

Build distribution:
>>> python setup.py sdist bdist_wheel

Notes
-----
This package requires Python 3.10+ and AVEVA Process Simulation.
Core implementation files are compiled to .pyc for distribution.
"""

import os

from setuptools import find_packages, setup


# Read the README file for long description
def read_readme():
    """Read README.md file for package description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "MOSKopt_Python"


setup(
    name="MOSKopt_Python",
    version="1.0.0",
    description="Stochastic Kriging based Optimization for AVEVA",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Tuse Asrav",
    author_email="tusas@kt.dtu.dk",
    packages=find_packages(include=["core", "examples", "simulation"]),
    include_package_data=True,
    package_data={
        "core": ["*.pyc"],  # Include compiled .pyc files
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "pyDOE>=0.3.8",
        "pandas>=1.1.0",
        "pyswarm>=0.6",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Optimization",
    ],
    keywords=[
        "optimization under uncertainty",
        "stochastic kriging",
        "gaussian-process",
        "bayesian-optimization",
        "chemical-engineering",
        "aveva",
        "process-simulation",
        "monte-carlo",
        "constraint-handling",
    ],
    platforms=["any"],
    license="Proprietary",
    zip_safe=False,
)
