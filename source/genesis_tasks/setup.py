"""Setup script for GenesisTasks package."""

from setuptools import find_packages, setup

# Read version from __init__.py or use default
__version__ = "0.1.0"

setup(
    name="genesis-tasks",
    version=__version__,
    description="Task definitions for GenesisLab",
    author="GenesisLab Developers",
    python_requires=">=3.9",
    packages=find_packages(where="genesis_tasks"),
    package_dir={"": "genesis_tasks"},
    install_requires=[
        "genesislab",  # GenesisLab framework
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
