#!/usr/bin/env python3
"""
DiLAC Hybrid WSD System
=======================

Setup script for installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="dilac-hybrid",
    version="2.0.0",
    author="DiLAC Team",
    author_email="dilac@example.com",
    description="Arabic lexical resource with hybrid WSD (DiLAC + AraBERT/CAMeLBERT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-repo]/dilac-hybrid",
    project_urls={
        "Bug Tracker": "https://github.com/[your-repo]/dilac-hybrid/issues",
        "Documentation": "https://dilac-hybrid.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Arabic",
    ],
    keywords=[
        "arabic",
        "nlp",
        "word-sense-disambiguation",
        "wsd",
        "semantic-similarity",
        "bert",
        "arabert",
        "camelbert",
        "lexical-resources",
        "dilac"
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "transformers": [
            "transformers>=4.20.0",
            "torch>=1.10.0",
            "datasets>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "transformers>=4.20.0",
            "torch>=1.10.0",
            "datasets>=2.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dilac-wsd=dilac.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dilac": ["data/*.json", "data/*.txt"],
    },
)
