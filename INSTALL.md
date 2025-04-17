# Installation Guide

## Prerequisites

- Python 3.8+
- pip

## Standard Installation

To install the latest released version:

```bash
git clone gitlabserv:/josq/heatdiff.git
cd heatdiff
pip install -e .
```

## Development Installation

To install in development mode with all dependencies:

```bash
git clone gitlabserv:/josq/heatdiff.git
cd heatdiff
pip install -e .[dev]
```

## Dependencies

The package requires:

- numpy
- matplotlib
- scipy

These will be installed automatically with the package.
