from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import DEFAULT_BUNDLE, DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT, DEFAULT_MODEL_BUNDLE_ROOT, resolve_path


DEFAULT_MODEL_ID = "cat-solv-47:xgboost"

LEARNER_LABELS = {
    "XGBoost": "xgboost",
    "RandomForest": "randomforest",
    "ExtraTrees": "extratrees",
    "HistGradientBoosting": "histgradientboosting",
}
LEARNER_FROM_LABEL = {value: key for key, value in LEARNER_LABELS.items()}
LAYER_LABELS = {
    "2D-Min": "2d-min",
    "2D-Elec": "2d-elec",
    "Sigma-Solv": "sigma-solv",
    "CAT-Solv-47": "cat-solv-47",
}
LAYER_FROM_LABEL = {value: key for key, value in LAYER_LABELS.items()}
LEARNER_ORDER = ["XGBoost", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
LAYER_ORDER = ["2D-Min", "2D-Elec", "Sigma-Solv", "CAT-Solv-47"]


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    learner: str
    feature_layer: str
    feature_count: int | None
    recommended: bool
    installed: bool
    bundle_path: Path
    recommended_use: str
    r2_mean: float | None = None
    mae_mean: float | None = None
    rmse_mean: float | None = None


def model_id_for(learner: str, feature_layer: str) -> str:
    if learner not in LEARNER_LABELS:
        raise KeyError(f"Unknown learner: {learner}")
    if feature_layer not in LAYER_LABELS:
        raise KeyError(f"Unknown feature layer: {feature_layer}")
    return f"{LAYER_LABELS[feature_layer]}:{LEARNER_LABELS[learner]}"


def parse_model_id(model_id: str) -> tuple[str, str]:
    try:
        layer_key, learner_key = model_id.split(":", 1)
        return LEARNER_FROM_LABEL[learner_key], LAYER_FROM_LABEL[layer_key]
    except Exception as exc:
        raise KeyError(f"Unknown model id: {model_id}") from exc


def bundle_path_for_model_id(model_id: str) -> Path:
    if model_id == DEFAULT_MODEL_ID:
        return DEFAULT_BUNDLE
    return DEFAULT_MODEL_BUNDLE_ROOT / model_id.replace(":", "__")


def _metrics_lookup() -> dict[tuple[str, str], dict[str, float]]:
    path = DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT / "outputs" / "aggregate_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        out[(str(row.learner), str(row.feature_layer))] = {
            "feature_count": int(row.feature_count),
            "R2_mean": float(row.R2_mean),
            "MAE_mean": float(row.MAE_mean),
            "RMSE_mean": float(row.RMSE_mean),
        }
    return out


def _metadata_model_id(path: Path) -> str | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return metadata.get("model_id")


def list_model_records() -> list[ModelRecord]:
    metrics = _metrics_lookup()
    records: list[ModelRecord] = []
    for learner in LEARNER_ORDER:
        for layer in LAYER_ORDER:
            model_id = model_id_for(learner, layer)
            bundle_path = bundle_path_for_model_id(model_id)
            installed = bundle_path.exists() and (bundle_path / "metadata.json").exists()
            if installed and _metadata_model_id(bundle_path) not in {None, model_id}:
                installed = False
            m = metrics.get((learner, layer), {})
            recommended = model_id == DEFAULT_MODEL_ID
            if recommended:
                use = "production/default CAT-Solv model"
            elif learner == "RandomForest" and layer == "CAT-Solv-47":
                use = "robustness/control only; not recommended for production"
            else:
                use = "robustness/control"
            records.append(
                ModelRecord(
                    model_id=model_id,
                    learner=learner,
                    feature_layer=layer,
                    feature_count=m.get("feature_count"),
                    recommended=recommended,
                    installed=installed,
                    bundle_path=bundle_path,
                    recommended_use=use,
                    r2_mean=m.get("R2_mean"),
                    mae_mean=m.get("MAE_mean"),
                    rmse_mean=m.get("RMSE_mean"),
                )
            )
    return records


def records_dataframe() -> pd.DataFrame:
    rows = []
    for record in list_model_records():
        rows.append(
            {
                "model_id": record.model_id,
                "learner": record.learner,
                "feature_layer": record.feature_layer,
                "feature_count": record.feature_count,
                "OOD_R2_mean": record.r2_mean,
                "OOD_MAE_mean": record.mae_mean,
                "OOD_RMSE_mean": record.rmse_mean,
                "recommended": record.recommended,
                "installed": record.installed,
                "recommended_use": record.recommended_use,
            }
        )
    return pd.DataFrame(rows)


def resolve_model_or_bundle(model: str | None = None, bundle: str | Path | None = None) -> Path:
    if bundle:
        return resolve_path(bundle)
    model_id = model or DEFAULT_MODEL_ID
    path = bundle_path_for_model_id(model_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{model_id}' is not installed at {path}. "
            f"Run `mica train-bundle --model {model_id} --output {path}` or choose an installed model."
        )
    return path
