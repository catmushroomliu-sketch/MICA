from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


def generate_binary_mixtures(
    solvent_csv: str | Path,
    output_csv: str | Path,
    ratios: list[float] | None = None,
    max_pairs: int | None = None,
) -> Path:
    ratios = ratios or [0.25, 0.5, 0.75]
    solvents = pd.read_csv(solvent_csv)
    required = {"name", "smiles"}
    missing = required - set(solvents.columns)
    if missing:
        raise ValueError(f"Solvent library is missing required columns: {sorted(missing)}")

    rows = []
    pair_iter = combinations(solvents.to_dict(orient="records"), 2)
    for pair_index, (a, b) in enumerate(pair_iter, start=1):
        if max_pairs is not None and pair_index > max_pairs:
            break
        for frac_a in ratios:
            frac_b = 1.0 - frac_a
            rows.append(
                {
                    "mixture_name": f"{a['name']}:{b['name']}={frac_a:.2f}:{frac_b:.2f}",
                    "solvent_a": a["name"],
                    "solvent_b": b["name"],
                    "smiles_a": a["smiles"],
                    "smiles_b": b["smiles"],
                    "fraction_a": frac_a,
                    "fraction_b": frac_b,
                    "mixture_smiles_proxy": f"{a['smiles']}.{b['smiles']}",
                    "modeling_note": "candidate mixture definition only; not a trained mixture-solvent prediction",
                }
            )
    out = Path(output_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def score_mixture_predictions(
    prediction_csv: str | Path,
    mixture_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    preds = pd.read_csv(prediction_csv)
    mixtures = pd.read_csv(mixture_csv)
    required_pred = {"SMILES_Solute", "SMILES_Solvent", "pred_logS_mean"}
    missing = required_pred - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")

    rows = []
    for mix in mixtures.itertuples(index=False):
        a = preds[preds["SMILES_Solvent"] == mix.smiles_a]
        b = preds[preds["SMILES_Solvent"] == mix.smiles_b]
        merged = a.merge(
            b,
            on="SMILES_Solute",
            suffixes=("_a", "_b"),
        )
        for row in merged.itertuples(index=False):
            pred = mix.fraction_a * row.pred_logS_mean_a + mix.fraction_b * row.pred_logS_mean_b
            rows.append(
                {
                    "SMILES_Solute": row.SMILES_Solute,
                    "mixture_name": mix.mixture_name,
                    "solvent_a": mix.solvent_a,
                    "solvent_b": mix.solvent_b,
                    "fraction_a": mix.fraction_a,
                    "fraction_b": mix.fraction_b,
                    "pred_logS_mixture_linear": pred,
                    "modeling_note": "linear blend of pure-solvent predictions; exploratory ranking only",
                }
            )
    out = Path(output_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out

