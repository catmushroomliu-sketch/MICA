# CLI Style

MICA follows a compact scientific-software command style inspired by established computational chemistry tools.

## Banner

```text
 __  __  ___   ____    _    
|  \/  ||_ _| / ___|  / \   
| |\/| | | | | |     / _ \  
| |  | | | | | |___ / ___ \ 
|_|  |_||___| \____/_/   \_\

Molecular Inference of Compatibility and Affinity
MICA 0.2.0 | high-throughput solubility inference
```

## Command grammar

```text
mica <command> [options]
```

Core commands:

| command | role |
|---|---|
| `doctor` | check Node/Python runtime readiness |
| `train-bundle` | train and export a model bundle |
| `inspect` | inspect a model bundle |
| `validate` | validate descriptor coverage |
| `run` | run batch inference |
| `screen` | rank candidate solvents |
| `solvent-library` | list/export solvent libraries |
| `mixture` | generate/score binary mixture candidates |
| `report` | generate compact HTML reports |

## Design rule

MICA should feel like a scientific executable, not a notebook script:

- no manuscript-generation commands in the public CLI
- explicit input validation before inference
- deterministic model-bundle metadata
- concise terminal output
- machine-readable CSV outputs
