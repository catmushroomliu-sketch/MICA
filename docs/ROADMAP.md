# MICA Scientific Software Roadmap

## Role

MICA is a scientific software implementation of the CAT-Solv model family. It is a prediction and screening engine, not a manuscript-writing utility.

## Current capabilities

| command | role |
|---|---|
| `train-bundle` | exports a reproducible model bundle |
| `models list` | lists production and robustness model definitions |
| `inspect` | inspects model metadata and descriptor requirements |
| `validate` | checks input descriptor compatibility |
| `domain` | audits row-level applicability-domain flags |
| `run` | performs high-throughput batch prediction |
| `screen` | ranks candidate solvents |
| `benchmark` | reports fixed solvent-OOD model metrics |
| `solvent-library` | lists/exports built-in solvent libraries |
| `mixture` | generates exploratory binary mixture candidates |
| `report` | writes compact HTML reports |

## Current limitation

The current top47 model requires prepared descriptor CSV files. This is intentional because sigma-moment descriptors cannot be faithfully inferred from plain SMILES without an external sigma/COSMO/xTB workflow or a validated descriptor library.

## Planned extensions

1. RDKit-only fast model for direct SMILES prediction.
2. Expand the built-in common-solvent descriptor library.
3. External xTB/COSMO feature-generation workflow.
4. SHAP-based local explanation output.
5. Optional SMILES-to-descriptor automation backed by an audited xTB/COSMO workflow.
6. GitHub release artifacts for full robustness model bundles.
