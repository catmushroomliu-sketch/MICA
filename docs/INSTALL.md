# Installation

MICA is distributed as an npm-installable command-line tool with a Python scientific-computing backend.

## Install from npm

Planned public package name:

```bash
npm install -g mica-solv
```

After installation:

```bash
mica doctor
mica --help
```

## Install from a local checkout

```bash
cd /Users/catmushroomliu/Documents/code/MICA
npm install -g .
mica doctor
```

## Python backend

The npm package provides the `mica` command and dispatches into the bundled Python package. The selected Python environment must contain:

```text
numpy
pandas
scikit-learn
xgboost
```

Set a specific Python interpreter if needed:

```bash
export MICA_PYTHON=/usr/bin/python3
mica doctor
```

## Basic use

```bash
mica inspect
mica validate --input examples/example_full98_features.csv
mica run --input examples/example_full98_features.csv --output examples/example_predictions.csv
mica screen --input examples/example_predictions.csv --output examples/example_screening.csv --top 20
```
