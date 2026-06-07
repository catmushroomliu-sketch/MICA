# MICA &mdash; Molecular Inference of Compatibility and Affinity

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Node.js >= 18](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org/)
[![Python >= 3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![version](https://img.shields.io/badge/version-0.2.0-lightgrey)](#)

**MICA** is a command-line toolkit for high-throughput **solubility prediction** and **solute&ndash;solvent compatibility screening**. It packages a validated 20-seed XGBoost ensemble trained on the CAT-Solv dataset into a reproducible, auditable CLI that scientists can run on their own machines.

```
 __  __  ___   ____    _
|  \/  ||_ _| / ___|  / \
| |\/| | | | | |     / _ \
| |  | | | | | |___ / ___ \
|_|  |_||___| \____/_/   \_\
```

---

## What MICA does

MICA takes a **prepared descriptor CSV** &mdash; rows of solute&ndash;solvent&ndash;temperature combinations with precomputed molecular features &mdash; and produces:

| output | description |
|---|---|
| **predictions** | logS (solubility) with ensemble mean, std, min, max per row |
| **screening** | per-solute solvent ranking by predicted solubility |
| **domain audit** | row-level applicability-domain flags (FAIL / WARN / PASS) |
| **mixture scoring** | binary solvent-mixture candidates with linear-blend scores |
| **HTML reports** | self-contained summary reports with domain summary &amp; data preview |

MICA **refuses to predict** when required descriptors are missing &mdash; it is designed to fail loudly rather than silently extrapolate.

---

## Quick start

```bash
# 1. Install
npm install -g mica-solv

# 2. Verify your environment
mica doctor

# 3. Predict solubility from a prepared descriptor CSV
mica run \
  --model cat-solv-47:xgboost \
  --input examples/example_full98_features.csv \
  --output predictions.csv

# 4. Rank solvents per solute
mica screen \
  --input predictions.csv \
  --output screening.csv \
  --top 10

# 5. Audit applicability domain
mica domain \
  --model cat-solv-47:xgboost \
  --input examples/example_full98_features.csv \
  --output domain_audit.csv

# 6. Generate an HTML report
mica report \
  --model cat-solv-47:xgboost \
  --input screening.csv \
  --domain domain_audit.csv \
  --output report.html
```

---

## Installation

### Prerequisites

- **Node.js** &ge; 18 &mdash; for the CLI wrapper
- **Python** &ge; 3.9 with:
  ```bash
  pip install numpy pandas scikit-learn xgboost
  ```

If you have multiple Python interpreters, set `MICA_PYTHON` to point MICA at the right one:

```bash
export MICA_PYTHON=/path/to/your/python3
```

### From npm

```bash
npm install -g mica-solv
mica doctor
```

### From source

```bash
git clone https://github.com/catmushroomliu-sketch/MICA.git
cd MICA
npm install -g .
mica doctor
```

### From PyPI

The Python package can also be installed directly:

```bash
pip install mica-solv
python -m mica --help
```

---

## Architecture

MICA distributes as a **dual package**:

| layer | technology | role |
|---|---|---|
| CLI shim | Node.js (`bin/mica.js`) | Finds a Python interpreter, prints the banner, dispatches all arguments to `python -m mica` |
| Core engine | Python (`mica/`) | All scientific logic: model loading, prediction, domain auditing, reporting |

The Node.js wrapper uses only Node built-ins (`child_process`, `fs`, `path`) &mdash; it has zero npm dependencies and simply delegates to the Python backend.

---

## The model: CAT-Solv top47 ensemble

The production model (`cat-solv-47:xgboost`) is a **20-seed XGBoost ensemble** where each seed is trained on an independently selected set of **47 molecular descriptors** (SHAP-ranked per seed). The union across seeds requires **52 unique features**.

### Benchmark performance (fixed 5-solvent OOD)

| metric | value |
|---|---|
| OOD R&sup2; | **0.788 &plusmn; 0.008** |
| OOD MAE | 0.413 &plusmn; 0.008 |
| OOD RMSE | 0.574 &plusmn; 0.011 |
| seeds | 20 (42&ndash;61) |
| held-out solvents | 5 (fixed split) |
| prediction unit | logS |

### Required features (52 total)

| category | count | examples |
|---|---|---|
| Solute RDKit descriptors | 18 | MolWt, TPSA, MolLogP, BalabanJ, &hellip; |
| Solute sigma-moments | 10 | sig2, sig3, pos_area_frac, neg_area_frac, HB_capacity, &hellip; |
| Solvent descriptors | 9 | LabuteASA, MolLogP, MaxPartialCharge, sigma-moments, &hellip; |
| Cross solute&ndash;solvent terms | 11 | HB_capacity_product, logP_abs_gap, sigma2_abs_gap, &hellip; |
| Class-conditioned terms | 3 | Class_ester_x_solute_HBD, &hellip; |
| Temperature | 1 | Temperature_K |

A complete feature list is available via `mica inspect --model cat-solv-47:xgboost`.

---

## Applicability domain

MICA audits every prediction row and assigns one of three statuses:

| flag | meaning |
|---|---|
| **PASS** | all features present, values within training ranges, known solvent &amp; temperature |
| **WARN** | features present but some values outside training min/max, unseen solvent, or temperature outside training range |
| **FAIL** | missing or NaN features &mdash; **prediction withheld** |

This is not a heuristic check. MICA loads the exact training feature ranges, solvent domain table, and required feature list from the model bundle and compares every row against them.

---

## CLI reference

| command | purpose |
|---|---|
| `mica doctor` | check runtime readiness (Node, Python, Python deps) |
| `mica models list` | list all 16 model definitions (4 learners &times; 4 feature layers) |
| `mica inspect` | print bundle metadata, features, and benchmark scores |
| `mica validate` | check whether a CSV satisfies bundle feature requirements |
| `mica run` | run ensemble prediction on a prepared CSV |
| `mica screen` | rank solvents per solute from prediction output |
| `mica domain` | audit applicability domain and write flags to CSV |
| `mica report` | generate a self-contained HTML report |
| `mica benchmark` | report OOD metrics from a bundle manifest |
| `mica solvent-library` | list or export the built-in solvent database |
| `mica mixture` | generate binary mixture candidates or score mixtures |
| `mica train-bundle` | train and export a reproducible model bundle |

Run `mica --help` or `mica <command> --help` for detailed usage.

---

## Built-in solvent library

MICA ships with a curated database of **31 common solvents** (`mica/data/solvent_library.csv`) including:

- **Categories**: aqueous, sulfoxide, lactam, amide, nitrile, alcohol, polyol, carbonate, ester, ketone, ether, aromatic/aliphatic hydrocarbons, halogenated, bio-based, lactone
- **Metadata**: SMILES, CAS numbers, boiling/flash points, polarity class, hydrogen-bond role, green &amp; industrial flags

```bash
# List all solvents
mica solvent-library

# Filter by category and export
mica solvent-library --category ester --output esters.csv
```

---

## Binary mixture scoring

MICA can generate and score binary solvent mixtures as an exploratory tool:

```bash
# Generate candidate mixtures
mica mixture generate \
  --library mica/data/solvent_library.csv \
  --ratios 0.25 0.5 0.75 \
  --output mixtures.csv

# Score mixtures from pure-solvent predictions
mica mixture score \
  --mixtures mixtures.csv \
  --predictions predictions.csv \
  --output mixture_scores.csv
```

Scores use a **linear blend** of pure-solvent logS predictions. This is intended as a screening heuristic, not a trained mixture model.

---

## Project structure

```
MICA/
  bin/mica.js               # Node.js CLI entry point
  mica/                     # Python package (the core engine)
    __init__.py
    __main__.py
    cli.py                  # argparse CLI (12 subcommands)
    bundle.py               # model bundle export / loading / training
    predict.py              # prediction, validation & screening engines
    domain.py               # applicability-domain auditing
    registry.py             # model registry (ID resolver, metadata)
    reporting.py            # compact HTML report generation
    paths.py                # filesystem path constants
    library.py              # built-in solvent library management
    mixture.py              # binary solvent mixture generation & scoring
    data/
      solvent_library.csv   # 31-solvent curated database
    reporting/              # (placeholder for future report templates)
  artifacts/
    mica_top47_bundle/      # shipped production model bundle
      models/               # 20 XGBoost .ubj files (seed_42..61)
      features/             # per-seed feature lists (JSON)
      required_features.json
      training_feature_ranges.csv
      solvent_domain.csv
      model_manifest.csv
      metadata.json
  examples/                 # example inputs and outputs
  benchmarks/               # benchmark harness & summary results
  docs/                     # developer documentation
  tests/                    # (placeholder)
  scripts/                  # (placeholder)
  package.json              # npm package metadata
  pyproject.toml            # Python package metadata
```

---

## Developer commands

```bash
# Clone and link for development
git clone https://github.com/catmushroomliu-sketch/MICA.git
cd MICA
npm link
mica doctor

# Run the Python module directly (bypasses Node)
python -m mica --help

# Smoke-test the production bundle
npm run smoke

# Train a fresh bundle (requires CAT-Solv experiment data)
python -m mica train-bundle --output artifacts/my_bundle

# Train a robustness-control model
python -m mica train-bundle \
  --model sigma-solv:extratrees \
  --output artifacts/mica_model_bundles/sigma-solv__extratrees
```

---

## Model registry

MICA defines **16 model configurations** (4 learners &times; 4 feature layers):

| feature layer | learners |
|---|---|
| 2D-Min | XGBoost, RandomForest, ExtraTrees, HistGradientBoosting |
| 2D-Elec | XGBoost, RandomForest, ExtraTrees, HistGradientBoosting |
| Sigma-Solv | XGBoost, RandomForest, ExtraTrees, HistGradientBoosting |
| CAT-Solv-47 | XGBoost (default), RandomForest, ExtraTrees, HistGradientBoosting |

Only the production model (`cat-solv-47:xgboost`) is shipped in the package. Other models serve as robustness controls and can be trained locally from the CAT-Solv experiment data.

---

## Design principles

- **Reproducible** &mdash; model bundles contain model files, per-seed feature lists, training ranges, metadata, and OOD benchmark checks
- **Explicit** &mdash; MICA refuses to predict when required descriptors are missing; no silent imputation
- **Domain-aware** &mdash; every row gets a FAIL / WARN / PASS flag based on training data boundaries
- **High-throughput** &mdash; ensemble prediction runs over arbitrary numbers of prepared rows
- **Interpretable-ready** &mdash; per-seed predictions and ensemble variance are exposed for downstream analysis
- **CLI-first** &mdash; designed to feel like a scientific executable, not a notebook script

---

## Limitations

- **No automatic SMILES featurization** &mdash; sigma-moment descriptor generation is deferred to a future release pending rigorous auditing. MICA expects pre-computed descriptor tables.
- **Mixure scoring is heuristic** &mdash; the linear-blend approximation is for ranking only; it is not a trained mixture model.
- **Temperature range** &mdash; the training data spans 283.15&ndash;323.15 K; predictions outside this range receive a WARN flag.
- **Solvent coverage** &mdash; the 5 held-out solvents represent a fixed OOD split; truly novel solvents may trigger domain warnings.

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned extensions including:

- RDKit-only fast screening model
- Expanded solvent descriptor library
- COSMO / xTB descriptor workflow
- SHAP explanation engine
- SMILES-to-descriptor automation
- GitHub release artifacts for robustness bundles

---

## Citation

If you use MICA in your research, please cite:

```bibtex
@software{mica_2026,
  author       = {Ziqi Liu},
  title        = {MICA: Molecular Inference of Compatibility and Affinity},
  year         = {2026},
  version      = {0.2.0},
  url          = {https://github.com/catmushroomliu-sketch/MICA},
  note         = {MIT License}
}
```

## License

[MIT](LICENSE) &copy; 2026 Ziqi Liu
