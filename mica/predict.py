from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .bundle import load_bundle, load_model_artifacts
from .paths import resolve_path


META_CANDIDATES = [
    "Common_RowID",
    "SMILES_Solute",
    "SMILES_Solvent",
    "Temperature_K",
    "LogS",
    "y_true",
    "solute_smiles",
    "solvent_smiles",
    "solute_id",
    "solvent_id",
]


def validate_input_features(df: pd.DataFrame, required_features: list[str]) -> list[str]:
    missing = [feature for feature in required_features if feature not in df.columns]
    return missing


def validate_csv(bundle_dir: str | Path, input_csv: str | Path) -> dict[str, object]:
    bundle = load_bundle(bundle_dir)
    input_path = resolve_path(input_csv)
    df = pd.read_csv(input_path, nrows=5)
    required = json.loads((bundle.root / "required_features.json").read_text(encoding="utf-8"))
    missing = validate_input_features(df, required)
    present = [feature for feature in required if feature in df.columns]
    return {
        "input": str(input_path),
        "bundle": str(bundle.root),
        "rows_checked": int(len(df)),
        "required_features": int(len(required)),
        "present_features": int(len(present)),
        "missing_features": int(len(missing)),
        "missing_feature_names": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def predict_csv(bundle_dir: str | Path, input_csv: str | Path, output_csv: str | Path) -> Path:
    bundle = load_bundle(bundle_dir)
    models = load_model_artifacts(bundle)
    input_path = resolve_path(input_csv)
    output_path = resolve_path(output_csv)
    df = pd.read_csv(input_path)

    required = json.loads((bundle.root / "required_features.json").read_text(encoding="utf-8"))
    missing = validate_input_features(df, required)
    if missing:
        preview = ", ".join(missing[:12])
        raise ValueError(
            f"Input CSV is missing {len(missing)} required features for the ensemble. "
            f"First missing features: {preview}"
        )

    pred_cols = []
    for seed, model, features, backend in models:
        col = f"pred_seed_{seed}"
        X = df[features].apply(pd.to_numeric, errors="coerce")
        if backend == "xgboost_booster":
            dm = xgb.DMatrix(X, feature_names=features)
            df[col] = model.predict(dm)
        elif backend == "sklearn_joblib":
            df[col] = model.predict(X.to_numpy(dtype=np.float32))
        else:
            raise ValueError(f"Unsupported model backend: {backend}")
        pred_cols.append(col)

    df["pred_logS_mean"] = df[pred_cols].mean(axis=1)
    df["pred_logS_std"] = df[pred_cols].std(axis=1)
    df["pred_logS_min"] = df[pred_cols].min(axis=1)
    df["pred_logS_max"] = df[pred_cols].max(axis=1)
    df["mica_model"] = bundle.metadata.get("name")
    df["mica_model_id"] = bundle.metadata.get("model_id", bundle.metadata.get("name"))
    df["mica_learner"] = bundle.metadata.get("learner", bundle.metadata.get("model_family"))
    df["mica_feature_layer"] = bundle.metadata.get("feature_layer", bundle.metadata.get("top_k"))
    df["mica_version"] = bundle.metadata.get("version")
    try:
        from .domain import domain_audit_dataframe

        domain_df = domain_audit_dataframe(bundle, df)
        for col in [
            "domain_status",
            "missing_feature_count",
            "nan_feature_count",
            "out_of_range_feature_count",
            "tail_feature_count",
            "temperature_status",
            "solvent_domain",
        ]:
            if col in domain_df.columns:
                df[col] = domain_df[col].to_numpy()
    except Exception:
        df["domain_status"] = "NOT_EVALUATED"

    meta_cols = [col for col in META_CANDIDATES if col in df.columns]
    domain_cols = [
        col
        for col in [
            "domain_status",
            "missing_feature_count",
            "nan_feature_count",
            "out_of_range_feature_count",
            "tail_feature_count",
            "temperature_status",
            "solvent_domain",
        ]
        if col in df.columns
    ]
    out_cols = meta_cols + [
        "pred_logS_mean",
        "pred_logS_std",
        "pred_logS_min",
        "pred_logS_max",
        "mica_model_id",
        "mica_learner",
        "mica_feature_layer",
        "mica_model",
        "mica_version",
    ] + domain_cols
    extra_cols = [col for col in df.columns if col.startswith("pred_seed_")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols + extra_cols].to_csv(output_path, index=False)
    return output_path


def screen_csv(prediction_csv: str | Path, output_csv: str | Path, top: int = 20) -> Path:
    input_path = resolve_path(prediction_csv)
    output_path = resolve_path(output_csv)
    df = pd.read_csv(input_path)
    if "pred_logS_mean" not in df.columns:
        raise ValueError("Prediction CSV must contain pred_logS_mean.")

    solute_col = "SMILES_Solute" if "SMILES_Solute" in df.columns else "solute_smiles"
    solvent_col = "SMILES_Solvent" if "SMILES_Solvent" in df.columns else "solvent_smiles"
    if solute_col not in df.columns or solvent_col not in df.columns:
        raise ValueError("Prediction CSV must contain solute and solvent SMILES columns for screening.")

    ranked = (
        df.sort_values([solute_col, "pred_logS_mean"], ascending=[True, False])
        .groupby(solute_col, as_index=False, group_keys=False)
        .head(top)
        .copy()
    )
    ranked["mica_rank"] = ranked.groupby(solute_col)["pred_logS_mean"].rank(method="first", ascending=False).astype(int)
    cols = [solute_col, solvent_col, "mica_rank", "pred_logS_mean"]
    if "pred_logS_std" in ranked.columns:
        cols.append("pred_logS_std")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked[cols].sort_values([solute_col, "mica_rank"]).to_csv(output_path, index=False)
    return output_path
