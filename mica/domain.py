from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import load_bundle
from .paths import resolve_path
from .predict import META_CANDIDATES, validate_input_features


def _load_required(bundle_root: Path) -> list[str]:
    return json.loads((bundle_root / "required_features.json").read_text(encoding="utf-8"))


def _load_ranges(bundle_root: Path) -> pd.DataFrame:
    path = bundle_root / "training_feature_ranges.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).set_index("feature")


def _load_solvent_domain(bundle_root: Path) -> pd.DataFrame:
    path = bundle_root / "solvent_domain.csv"
    if path.exists():
        return pd.read_csv(path)
    ood_path = bundle_root / "ood_solvents.csv"
    if not ood_path.exists():
        return pd.DataFrame()
    ood = pd.read_csv(ood_path)
    solvent_col = "SMILES_Solvent" if "SMILES_Solvent" in ood.columns else "smiles"
    name_col = "Solvent_Name" if "Solvent_Name" in ood.columns else "name"
    return pd.DataFrame(
        {
            "SMILES_Solvent": ood[solvent_col],
            "Solvent_Name": ood[name_col],
            "domain_role": "fixed_ood",
        }
    )


def _classify_solvents(df: pd.DataFrame, domain: pd.DataFrame) -> pd.Series:
    if "SMILES_Solvent" not in df.columns:
        return pd.Series(["missing_solvent_column"] * len(df), index=df.index)
    if domain.empty or "SMILES_Solvent" not in domain.columns:
        return pd.Series(["unknown_reference"] * len(df), index=df.index)
    mapping = dict(zip(domain["SMILES_Solvent"].astype(str), domain["domain_role"].astype(str)))
    return df["SMILES_Solvent"].astype(str).map(mapping).fillna("unknown")


def _temperature_status(df: pd.DataFrame, ranges: pd.DataFrame) -> pd.Series:
    if "Temperature_K" not in df.columns:
        return pd.Series(["missing_temperature_column"] * len(df), index=df.index)
    if ranges.empty or "Temperature_K" not in ranges.index:
        return pd.Series(["unknown_reference"] * len(df), index=df.index)
    low = float(ranges.loc["Temperature_K", "min"])
    high = float(ranges.loc["Temperature_K", "max"])
    temp = pd.to_numeric(df["Temperature_K"], errors="coerce")
    return pd.Series(
        np.where(temp.isna(), "missing", np.where((temp < low) | (temp > high), "outside_training_range", "within_training_range")),
        index=df.index,
    )


def domain_audit(bundle_dir: str | Path, input_csv: str | Path, output_csv: str | Path) -> Path:
    bundle = load_bundle(bundle_dir)
    input_path = resolve_path(input_csv)
    output_path = resolve_path(output_csv)
    df = pd.read_csv(input_path)
    out = domain_audit_dataframe(bundle, df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def domain_audit_dataframe(bundle, df: pd.DataFrame) -> pd.DataFrame:
    required = _load_required(bundle.root)
    ranges = _load_ranges(bundle.root)
    solvent_domain = _load_solvent_domain(bundle.root)

    missing = validate_input_features(df, required)
    present = [feature for feature in required if feature in df.columns]
    out = df[[col for col in META_CANDIDATES if col in df.columns]].copy()
    out["required_feature_count"] = len(required)
    out["missing_feature_count"] = len(missing)
    out["missing_feature_names"] = ";".join(missing[:50])

    if present:
        numeric = df[present].apply(pd.to_numeric, errors="coerce")
        out["nan_feature_count"] = numeric.isna().sum(axis=1).astype(int)
    else:
        numeric = pd.DataFrame(index=df.index)
        out["nan_feature_count"] = 0

    out_of_range_counts = np.zeros(len(df), dtype=int)
    tail_counts = np.zeros(len(df), dtype=int)
    out_feature_names: list[list[str]] = [[] for _ in range(len(df))]
    if not ranges.empty:
        for feature in present:
            if feature not in ranges.index:
                continue
            values = pd.to_numeric(df[feature], errors="coerce")
            row = ranges.loc[feature]
            outside = (values < float(row["min"])) | (values > float(row["max"]))
            tail = (values < float(row["p01"])) | (values > float(row["p99"]))
            outside = outside.fillna(False).to_numpy()
            tail = tail.fillna(False).to_numpy()
            out_of_range_counts += outside.astype(int)
            tail_counts += tail.astype(int)
            if outside.any():
                for idx in np.flatnonzero(outside):
                    if len(out_feature_names[idx]) < 20:
                        out_feature_names[idx].append(feature)
    out["out_of_range_feature_count"] = out_of_range_counts
    out["tail_feature_count"] = tail_counts
    out["out_of_range_features"] = [";".join(items) for items in out_feature_names]
    out["temperature_status"] = _temperature_status(df, ranges)
    out["solvent_domain"] = _classify_solvents(df, solvent_domain)

    fail = (out["missing_feature_count"] > 0) | (out["nan_feature_count"] > 0)
    warn = (
        (out["out_of_range_feature_count"] > 0)
        | out["temperature_status"].isin(["outside_training_range", "missing", "missing_temperature_column"])
        | out["solvent_domain"].isin(["unknown", "missing_solvent_column"])
    )
    out["domain_status"] = np.where(fail, "FAIL", np.where(warn, "WARN", "PASS"))
    out["mica_model_id"] = bundle.metadata.get("model_id", bundle.metadata.get("name"))
    out["mica_learner"] = bundle.metadata.get("learner", bundle.metadata.get("model_family"))
    out["mica_feature_layer"] = bundle.metadata.get("feature_layer", bundle.metadata.get("top_k"))
    return out


def domain_summary(domain_csv: str | Path) -> dict[str, object]:
    df = pd.read_csv(domain_csv)
    return {
        "rows": int(len(df)),
        "status_counts": df["domain_status"].value_counts().to_dict() if "domain_status" in df.columns else {},
        "solvent_domain_counts": df["solvent_domain"].value_counts().to_dict() if "solvent_domain" in df.columns else {},
        "mean_nan_feature_count": float(df["nan_feature_count"].mean()) if "nan_feature_count" in df.columns else None,
        "mean_out_of_range_feature_count": float(df["out_of_range_feature_count"].mean()) if "out_of_range_feature_count" in df.columns else None,
    }
