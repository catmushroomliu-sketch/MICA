from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .paths import DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT, DEFAULT_PHYSCHEM_EXPERIMENT, DEFAULT_TOPK_EXPERIMENT, resolve_path
from .registry import DEFAULT_MODEL_ID, model_id_for


MODEL_NAME = "mica_top47_valshap_ensemble"
MODEL_VERSION = "0.2.0"
DEFAULT_SEEDS = list(range(42, 62))


@dataclass(frozen=True)
class Bundle:
    root: Path
    metadata: dict


def load_bundle(bundle_dir: str | Path) -> Bundle:
    root = resolve_path(bundle_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("name") == MODEL_NAME and "model_id" not in metadata:
        metadata = {
            **metadata,
            "model_id": DEFAULT_MODEL_ID,
            "learner": "XGBoost",
            "feature_layer": "CAT-Solv-47",
            "feature_count": int(metadata.get("top_k", 47)),
            "backend": "xgboost_booster",
            "recommended_use": "production/default CAT-Solv model",
            "endpoint": "temperature-dependent experimental logS for prepared solute-solvent descriptor rows",
            "algorithm": "20-seed XGBoost ensemble with seed-specific validation-SHAP top-47 features",
            "ood_protocol": metadata.get("benchmark", {}).get("ood_protocol", "original-five leave-solvent-out, 20 seeds"),
            "domain_definition": "Prepared descriptor CSV with required features; row-level warnings for missing values, temperature outside training range, unseen solvents, and descriptors outside training min/max.",
        }
    return Bundle(root=root, metadata=metadata)


def _import_physchem_helpers() -> object:
    sys.path.insert(0, str(DEFAULT_PHYSCHEM_EXPERIMENT))
    import run_physchem_cross_20seed as helpers  # type: ignore

    return helpers


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _feature_ranges(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "feature": feature,
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "median": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "non_null_count": 0,
                }
            )
            continue
        rows.append(
            {
                "feature": feature,
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "non_null_count": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _solvent_domain_table(
    df: pd.DataFrame,
    remain_row_ids: list[int] | set[int],
    test_row_ids: list[int] | set[int],
    solvent_names: dict[str, str],
) -> pd.DataFrame:
    rows = []
    train = df[df["Common_RowID"].isin(remain_row_ids)]
    test = df[df["Common_RowID"].isin(test_row_ids)]
    train_counts = train["SMILES_Solvent"].value_counts()
    test_counts = test["SMILES_Solvent"].value_counts()
    solvents = sorted(set(train_counts.index.astype(str)) | set(test_counts.index.astype(str)))
    for smiles in solvents:
        train_n = int(train_counts.get(smiles, 0))
        test_n = int(test_counts.get(smiles, 0))
        role = "fixed_ood" if test_n > 0 and train_n == 0 else "training"
        rows.append(
            {
                "SMILES_Solvent": smiles,
                "Solvent_Name": solvent_names.get(smiles, smiles),
                "domain_role": role,
                "training_rows": train_n,
                "fixed_ood_rows": test_n,
            }
        )
    return pd.DataFrame(rows).sort_values(["domain_role", "Solvent_Name"]).reset_index(drop=True)


def _write_domain_assets(
    output: Path,
    df: pd.DataFrame,
    required_features: list[str],
    remain_row_ids,
    test_row_ids,
    solvent_names: dict[str, str],
) -> None:
    train_pool = df[df["Common_RowID"].isin(remain_row_ids)]
    _feature_ranges(train_pool, required_features).to_csv(output / "training_feature_ranges.csv", index=False)
    _solvent_domain_table(df, remain_row_ids, test_row_ids, solvent_names).to_csv(output / "solvent_domain.csv", index=False)


def _selected_features_for_seed(selected_df: pd.DataFrame, seed: int, top_k: int = 47) -> list[str]:
    sub = selected_df[(selected_df["seed"] == seed) & (selected_df["top_k"] == top_k)].sort_values("rank")
    features = sub["feature"].tolist()
    if len(features) != top_k:
        raise RuntimeError(f"Seed {seed} expected {top_k} features, found {len(features)}.")
    return features


def export_top47_bundle(
    output_dir: str | Path,
    topk_experiment: str | Path = DEFAULT_TOPK_EXPERIMENT,
    force: bool = False,
) -> Path:
    output = resolve_path(output_dir)
    if output.exists():
        if not force:
            raise FileExistsError(f"Bundle already exists: {output}. Use --force to overwrite.")
        shutil.rmtree(output)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(parents=True, exist_ok=True)

    topk_root = resolve_path(topk_experiment)
    selected_path = topk_root / "outputs" / "selected_features_by_seed.csv"
    aggregate_path = topk_root / "outputs" / "aggregate_seed_metrics.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected features file: {selected_path}")

    helpers = _import_physchem_helpers()
    frames, _ = helpers.build_model_frames()
    df = frames["v2_physchem_class_cross"].copy()
    ood_table, remain_row_ids, test_row_ids = helpers.build_test_manifest(frames["v1"])
    selected_df = pd.read_csv(selected_path)

    manifest_rows = []
    prediction_frames = []
    for seed in DEFAULT_SEEDS:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)
        idx_train = helpers.idx_from_row_ids(df, sorted(train_ids))
        idx_val = helpers.idx_from_row_ids(df, sorted(val_ids))
        idx_test = helpers.idx_from_row_ids(df, test_row_ids)

        features = _selected_features_for_seed(selected_df, seed, top_k=47)
        X = df[features]
        y = df["LogS"].to_numpy()
        model = helpers.fit_xgb(
            seed,
            X.iloc[idx_train].to_numpy(),
            y[idx_train],
            X.iloc[idx_val].to_numpy(),
            y[idx_val],
        )
        model_path = output / "models" / f"seed_{seed}.ubj"
        feature_path = output / "features" / f"seed_{seed}_features.json"
        model.save_model(model_path)
        feature_path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")

        y_true = y[idx_test]
        y_pred = model.predict(X.iloc[idx_test].to_numpy())
        manifest_rows.append(
            {
                "seed": seed,
                "model_file": str(model_path.relative_to(output)),
                "feature_file": str(feature_path.relative_to(output)),
                "backend": "xgboost_booster",
                "learner": "XGBoost",
                "feature_layer": "CAT-Solv-47",
                "feature_count": len(features),
                **_metrics(y_true, y_pred),
            }
        )
        pred_df = df.iloc[idx_test][helpers.PRED_META_COLS].copy()
        pred_df = pred_df.rename(columns={"LogS": "y_true"})
        pred_df["seed"] = seed
        pred_df["y_pred"] = y_pred
        prediction_frames.append(pred_df)
        print(f"exported seed {seed}", flush=True)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output / "model_manifest.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "ood_predictions_bundle_check.csv", index=False)
    ood_table.to_csv(output / "ood_solvents.csv", index=False)

    all_required = sorted(set().union(*[
        set(json.loads((output / row["feature_file"]).read_text(encoding="utf-8")))
        for row in manifest_rows
    ]))
    (output / "required_features.json").write_text(json.dumps(all_required, indent=2) + "\n", encoding="utf-8")
    _write_domain_assets(output, df, all_required, remain_row_ids, test_row_ids, helpers.SOLVENT_NAMES)

    aggregate = pd.read_csv(aggregate_path) if aggregate_path.exists() else pd.DataFrame()
    metadata = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "model_id": DEFAULT_MODEL_ID,
        "description": "20-seed validation-SHAP top47 XGBoost ensemble for CAT-Solv/MICA.",
        "learner": "XGBoost",
        "feature_layer": "CAT-Solv-47",
        "feature_count": 47,
        "backend": "xgboost_booster",
        "recommended_use": "production/default CAT-Solv model",
        "endpoint": "temperature-dependent experimental logS for prepared solute-solvent descriptor rows",
        "algorithm": "20-seed XGBoost ensemble with seed-specific validation-SHAP top-47 features",
        "ood_protocol": "original-five leave-solvent-out, 20 seeds",
        "domain_definition": "Prepared descriptor CSV with required features; row-level warnings for missing values, temperature outside training range, unseen solvents, and descriptors outside training min/max.",
        "model_family": "XGBoostRegressor ensemble",
        "top_k": 47,
        "seeds": DEFAULT_SEEDS,
        "n_models": len(DEFAULT_SEEDS),
        "prediction_units": "logS",
        "feature_mode": "prepared_full98_descriptor_csv",
        "required_unique_feature_count": len(all_required),
        "source_topk_experiment": str(topk_root),
        "source_physchem_experiment": str(DEFAULT_PHYSCHEM_EXPERIMENT),
        "benchmark": {
            "ood_protocol": "original-five leave-solvent-out, 20 seeds",
            "reported_top47_r2_mean": 0.7883833939332014,
            "reported_top47_r2_std": 0.008127044864693633,
            "bundle_check_r2_mean": float(manifest_df["R2"].mean()),
            "bundle_check_r2_std": float(manifest_df["R2"].std(ddof=1)),
        },
    }
    if not aggregate.empty:
        metadata["source_aggregate_metrics"] = aggregate.to_dict(orient="records")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output


def _import_learner_helpers() -> object:
    sys.path.insert(0, str(DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT))
    import run_learner_robustness_20seed as helpers  # type: ignore

    return helpers


def export_model_bundle(
    output_dir: str | Path,
    learner: str = "XGBoost",
    feature_layer: str = "CAT-Solv-47",
    force: bool = False,
) -> Path:
    """Export one learner/layer 20-seed bundle using the fixed solvent-OOD protocol."""
    if learner == "XGBoost" and feature_layer == "CAT-Solv-47":
        return export_top47_bundle(output_dir, force=force)

    helpers = _import_learner_helpers()
    if learner not in helpers.LEARNERS:
        raise ValueError(f"Unknown learner: {learner}. Choices: {', '.join(helpers.LEARNERS)}")
    if feature_layer not in helpers.FEATURE_LAYERS:
        raise ValueError(f"Unknown feature layer: {feature_layer}. Choices: {', '.join(helpers.FEATURE_LAYERS)}")

    output = resolve_path(output_dir)
    if output.exists():
        if not force:
            raise FileExistsError(f"Bundle already exists: {output}. Use --force to overwrite.")
        shutil.rmtree(output)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(parents=True, exist_ok=True)

    frames, _ = helpers.build_model_frames()
    frames = helpers.add_2d_min_frame(frames)
    top47_by_seed = helpers.load_top47_by_seed()
    ood_table, remain_row_ids, test_row_ids = helpers.build_test_manifest(frames["v1"])
    df = frames[helpers.LAYER_TO_FRAME[feature_layer]].copy()

    manifest_rows = []
    prediction_frames = []
    required_sets = []
    for seed in helpers.SEEDS:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)
        train_idx = helpers.idx_from_row_ids(df, sorted(train_ids))
        val_idx = helpers.idx_from_row_ids(df, sorted(val_ids))
        test_idx = helpers.idx_from_row_ids(df, test_row_ids)
        features = helpers.layer_feature_columns(df, feature_layer, seed, top47_by_seed)
        required_sets.append(set(features))

        X = df[features].to_numpy(dtype=np.float32)
        y = df["LogS"].to_numpy(dtype=np.float32)
        model = helpers.FITTERS[learner](seed, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        if learner == "XGBoost":
            backend = "xgboost_booster"
            model_path = output / "models" / f"seed_{seed}.ubj"
            model.save_model(model_path)
        else:
            backend = "sklearn_joblib"
            model_path = output / "models" / f"seed_{seed}.joblib"
            joblib.dump(model, model_path)

        feature_path = output / "features" / f"seed_{seed}_features.json"
        feature_path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
        y_true = y[test_idx]
        y_pred = model.predict(X[test_idx])
        manifest_rows.append(
            {
                "seed": seed,
                "model_file": str(model_path.relative_to(output)),
                "feature_file": str(feature_path.relative_to(output)),
                "backend": backend,
                "learner": learner,
                "feature_layer": feature_layer,
                "feature_count": len(features),
                **_metrics(y_true, y_pred),
            }
        )
        pred_df = df.iloc[test_idx][helpers.PRED_META_COLS].copy()
        pred_df = pred_df.rename(columns={"LogS": "y_true"})
        pred_df["seed"] = seed
        pred_df["y_pred"] = y_pred
        prediction_frames.append(pred_df)
        print(f"exported {learner} {feature_layer} seed {seed}", flush=True)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output / "model_manifest.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "ood_predictions_bundle_check.csv", index=False)
    ood_table.to_csv(output / "ood_solvents.csv", index=False)

    all_required = sorted(set().union(*required_sets))
    (output / "required_features.json").write_text(json.dumps(all_required, indent=2) + "\n", encoding="utf-8")
    _write_domain_assets(output, df, all_required, remain_row_ids, test_row_ids, helpers.SOLVENT_NAMES)

    aggregate_path = DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT / "outputs" / "aggregate_metrics.csv"
    aggregate = pd.read_csv(aggregate_path) if aggregate_path.exists() else pd.DataFrame()
    aggregate_row = aggregate[(aggregate["learner"].eq(learner)) & (aggregate["feature_layer"].eq(feature_layer))]
    model_id = model_id_for(learner, feature_layer)
    metadata = {
        "name": model_id.replace(":", "_"),
        "version": MODEL_VERSION,
        "model_id": model_id,
        "description": f"20-seed {learner} {feature_layer} robustness bundle for CAT-Solv/MICA.",
        "learner": learner,
        "feature_layer": feature_layer,
        "feature_count": int(manifest_df["feature_count"].iloc[0]),
        "backend": str(manifest_df["backend"].iloc[0]),
        "recommended_use": "robustness/control only",
        "endpoint": "temperature-dependent experimental logS for prepared solute-solvent descriptor rows",
        "algorithm": f"20-seed {learner} ensemble using the {feature_layer} descriptor layer",
        "ood_protocol": "original-five leave-solvent-out, 20 seeds",
        "domain_definition": "Prepared descriptor CSV with required features; row-level warnings for missing values, temperature outside training range, unseen solvents, and descriptors outside training min/max.",
        "model_family": f"{learner} ensemble",
        "seeds": [int(x) for x in helpers.SEEDS],
        "n_models": int(len(helpers.SEEDS)),
        "prediction_units": "logS",
        "feature_mode": "prepared_descriptor_csv",
        "required_unique_feature_count": len(all_required),
        "source_learner_robustness_experiment": str(DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT),
        "benchmark": {
            "ood_protocol": "original-five leave-solvent-out, 20 seeds",
            "bundle_check_r2_mean": float(manifest_df["R2"].mean()),
            "bundle_check_r2_std": float(manifest_df["R2"].std(ddof=1)),
            "bundle_check_mae_mean": float(manifest_df["MAE"].mean()),
            "bundle_check_rmse_mean": float(manifest_df["RMSE"].mean()),
        },
    }
    if not aggregate_row.empty:
        metadata["source_aggregate_metrics"] = aggregate_row.to_dict(orient="records")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output


def load_model_artifacts(bundle: Bundle) -> list[tuple[int, object, list[str], str]]:
    manifest = pd.read_csv(bundle.root / "model_manifest.csv")
    artifacts: list[tuple[int, object, list[str], str]] = []
    for row in manifest.itertuples(index=False):
        backend = getattr(row, "backend", None) or bundle.metadata.get("backend") or "xgboost_booster"
        features = json.loads((bundle.root / row.feature_file).read_text(encoding="utf-8"))
        if backend == "xgboost_booster":
            model = xgb.Booster()
            model.load_model(str(bundle.root / row.model_file))
        elif backend == "sklearn_joblib":
            model = joblib.load(bundle.root / row.model_file)
        else:
            raise ValueError(f"Unsupported model backend: {backend}")
        artifacts.append((int(row.seed), model, features, str(backend)))
    return artifacts


def load_models(bundle: Bundle) -> list[tuple[int, xgb.Booster, list[str]]]:
    manifest = pd.read_csv(bundle.root / "model_manifest.csv")
    models: list[tuple[int, xgb.Booster, list[str]]] = []
    for row in manifest.itertuples(index=False):
        booster = xgb.Booster()
        booster.load_model(str(bundle.root / row.model_file))
        features = json.loads((bundle.root / row.feature_file).read_text(encoding="utf-8"))
        models.append((int(row.seed), booster, features))
    return models
