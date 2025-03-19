import os
from setuptools import setup, find_packages

long_description = ""
if os.path.isfile("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="MolTransformer",
    version="0.2.1",
    author="Chih-Hsuan Yang",
    author_email="chyang@iastate.edu",
    description="A python package for the paper: MolGen-Transformer: A molecule language model for the generation and latent space exploration of pi-conjugated molecules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaskarGroup/MolTransformer_repo",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10",
        "numpy>=1.19",
        "matplotlib>=3.0",
        "pillow>=9.0",
        "pandas>=1.0",
        "scikit-learn>=0.24",
        "scipy>=1.5",
        "selfies>=2.0",
        "certifi>=2021.5.30",
        "pubchempy>=1.0",
        "hostlist",
        # "rdkit-pypi>=2022.9.5"  # Unofficial PyPI package for RDKit
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
