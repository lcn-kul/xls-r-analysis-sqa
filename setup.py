import os
from setuptools import find_packages, setup

# Read the README for PyPI long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="xls_r_sqa",
    version="0.1.0",
    author="Bastiaan Tamm",
    author_email="bastiaan.tamm@kuleuven.be",
    description="Models for the paper 'Analysis of XLS-R for Speech Quality Assessment'.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lcn-kul/xls-r-analysis-sqa",
    license="MIT",
    packages=find_packages(),  # finds xls_r_sqa and any subpackages
    include_package_data=True,  # include non-.py files like model checkpoints
    python_requires=">=3.6",
    install_requires=[
        "librosa>=0.9",
        "numpy>=1.23",
        "soundfile>=0.11",
        "torchaudio>=0.11",
        "torch>=1.11",
        "transformers>=4.25",
    ],
)
