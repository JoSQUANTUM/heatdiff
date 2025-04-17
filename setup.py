from setuptools import find_packages, setup

setup(
    name="heatdiff",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "flake8>=4.0",
            "ruff>=0.11.4",
            "isort>=5.0",
        ],
    },
)
