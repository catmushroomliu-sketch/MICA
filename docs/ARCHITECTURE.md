# MICA Architecture

## Purpose

MICA is a command-line scientific software tool for molecular inference of solute-solvent compatibility and solubility. The design goal is reproducible, high-throughput prediction from explicit molecular descriptor tables.

The first release packages the validated CAT-Solv top47 ensemble as a rigorous inference engine. It does not hide missing quantum/sigma descriptors behind uncontrolled approximations.

## Layers

| layer | responsibility |
|---|---|
| model registry | maps stable model IDs to installed bundles and benchmark metadata |
| model bundle | stores trained models, per-seed feature lists, domain assets, and metadata |
| prediction engine | validates feature columns, runs ensemble prediction, reports mean/std |
| screening engine | ranks solvents for each solute from prediction CSVs |
| validation engine | checks descriptor availability before model execution |
| domain engine | audits missing values, descriptor ranges, temperature range, and solvent-domain role |
| feature generation | reserved for stage 2; current top47 requires prepared descriptor CSV |

## Current model

The default first-stage model is a 20-seed ensemble reproducing the validated `class_cross_top47_valshap` setting:

- OOD protocol: original-five leave-solvent-out benchmark
- feature source: `v2_physchem_class_cross`
- feature selection: validation-SHAP ranking inside the training pool
- per-seed features: top 47 selected independently for each seed
- reported benchmark: OOD R2 = 0.7884 +/- 0.0081

The stable production model ID is `cat-solv-47:xgboost`. Other tree learners and descriptor layers are available as robustness/control definitions through `mica models list` and can be exported as separate bundles.

## Applicability domain

Each v0.2 bundle should include:

- `required_features.json`
- `training_feature_ranges.csv`
- `solvent_domain.csv`
- `model_manifest.csv`
- `metadata.json`

The `mica domain` command uses these files to produce row-level `PASS/WARN/FAIL` flags. This is intentionally explicit: MICA does not silently extrapolate beyond missing descriptors, unknown solvents, or descriptor ranges without reporting it.

## Why not automatic SMILES-only prediction in stage 1?

The best CAT-Solv model depends on sigma-moment descriptors. These are not ordinary RDKit descriptors and cannot be faithfully generated from plain SMILES without an external sigma/COSMO/xTB pipeline or a precomputed solvent descriptor library.

The defensible first release is therefore:

1. package the validated model,
2. predict prepared feature tables at high throughput,
3. expose model requirements clearly,
4. add full SMILES-to-sigma automation in a later release.

## Command model

| command | role |
|---|---|
| `doctor` | check runtime and dependency readiness |
| `models list` | list model registry entries and installed status |
| `train-bundle` | export a reproducible model bundle |
| `inspect` | inspect a bundle |
| `validate` | check input descriptor coverage |
| `domain` | audit applicability-domain flags |
| `run` | run predictions |
| `screen` | rank candidate solvents |
| `benchmark` | reproduce fixed solvent-OOD bundle metrics |
