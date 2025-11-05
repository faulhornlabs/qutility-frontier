from setuptools import find_packages, setup

setup(
    name="scalable_volumetric_benchmark",
    version="0.1.0",
    description="Quantum circuit utilities for volumetric benchmarking.",
    packages=find_packages(include=["benchmarks", "benchmarks.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
    ],
)
