"""Setup script for GenesisLab package."""

from setuptools import find_packages, setup

# Read version from __init__.py or use default
__version__ = "0.1.0"

setup(
    name="genesislab",
    version=__version__,
    description="Manager-based RL framework for Genesis physics engine",
    author="GenesisLab Developers",
    python_requires=">=3.9",
    packages=find_packages(where="genesislab"),
    package_dir={"": "genesislab"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "genesis",  # Assume Genesis is installed separately
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
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
