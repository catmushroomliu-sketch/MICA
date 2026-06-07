from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from .bundle import export_model_bundle, export_top47_bundle, load_bundle
from .domain import domain_audit
from .library import export_library, filter_library
from .mixture import generate_binary_mixtures, score_mixture_predictions
from .predict import predict_csv, screen_csv, validate_csv
from .reporting import write_screen_report
from .registry import DEFAULT_MODEL_ID, bundle_path_for_model_id, model_id_for, parse_model_id, records_dataframe, resolve_model_or_bundle
from .paths import DEFAULT_BUNDLE


def cmd_fit_bundle(args: argparse.Namespace) -> None:
    if args.model:
        learner, layer = parse_model_id(args.model)
    else:
        learner = args.learner
        layer = args.layer
    model_id = args.model or model_id_for(learner, layer)
    output = args.output or bundle_path_for_model_id(model_id)
    if learner == "XGBoost" and layer == "CAT-Solv-47":
        path = export_top47_bundle(output, force=args.force)
    else:
        path = export_model_bundle(output, learner=learner, feature_layer=layer, force=args.force)
    print(f"MICA bundle exported: {path}")


def cmd_models(args: argparse.Namespace) -> None:
    df = records_dataframe()
    if args.models_command == "list":
        view = df.copy()
        numeric_cols = ["OOD_R2_mean", "OOD_MAE_mean", "OOD_RMSE_mean"]
        for col in numeric_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        print(view.to_string(index=False))
        return
    raise SystemExit("Missing models subcommand.")


def cmd_info(args: argparse.Namespace) -> None:
    bundle = load_bundle(resolve_model_or_bundle(args.model, args.bundle))
    required = json.loads((bundle.root / "required_features.json").read_text(encoding="utf-8"))
    print(f"model_id: {bundle.metadata.get('model_id', bundle.metadata.get('name'))}")
    print(f"name: {bundle.metadata.get('name')}")
    print(f"version: {bundle.metadata.get('version')}")
    print(f"learner: {bundle.metadata.get('learner')}")
    print(f"feature_layer: {bundle.metadata.get('feature_layer')}")
    print(f"feature_count: {bundle.metadata.get('feature_count')}")
    print(f"models: {bundle.metadata.get('n_models')}")
    print(f"required unique features: {len(required)}")
    print(f"endpoint: {bundle.metadata.get('endpoint')}")
    print(f"algorithm: {bundle.metadata.get('algorithm')}")
    print(f"domain_definition: {bundle.metadata.get('domain_definition')}")
    print("benchmark:")
    for key, value in bundle.metadata.get("benchmark", {}).items():
        print(f"  {key}: {value}")


def cmd_predict_csv(args: argparse.Namespace) -> None:
    path = predict_csv(resolve_model_or_bundle(args.model, args.bundle), args.input, args.output)
    print(f"Predictions written: {path}")


def cmd_screen_csv(args: argparse.Namespace) -> None:
    path = screen_csv(args.input, args.output, top=args.top)
    print(f"Screening table written: {path}")


def cmd_validate(args: argparse.Namespace) -> None:
    result = validate_csv(resolve_model_or_bundle(args.model, args.bundle), args.input)
    print(f"status: {result['status']}")
    print(f"input: {result['input']}")
    print(f"required_features: {result['required_features']}")
    print(f"present_features: {result['present_features']}")
    print(f"missing_features: {result['missing_features']}")
    missing = result["missing_feature_names"]
    if missing:
        print("missing_feature_names:")
        for name in missing[:50]:
            print(f"  {name}")
        if len(missing) > 50:
            print(f"  ... {len(missing) - 50} more")
        raise SystemExit(1)


def cmd_doctor(args: argparse.Namespace) -> None:
    modules = ["numpy", "pandas", "sklearn", "xgboost"]
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    print(f"python: {sys.executable}")
    print(f"default_bundle: {DEFAULT_BUNDLE}")
    print(f"default_bundle_exists: {DEFAULT_BUNDLE.exists()}")
    print(f"python_deps: {'PASS' if not missing else 'FAIL'}")
    if missing:
        print("missing:")
        for name in missing:
            print(f"  {name}")
        raise SystemExit(1)


def cmd_domain(args: argparse.Namespace) -> None:
    path = domain_audit(resolve_model_or_bundle(args.model, args.bundle), args.input, args.output)
    print(f"Domain audit written: {path}")


def cmd_solvent_library(args: argparse.Namespace) -> None:
    if args.library_command == "list":
        df = filter_library(
            category=args.category,
            polarity=args.polarity,
            green_only=args.green_only,
            industrial_only=args.industrial_only,
        )
        cols = ["name", "smiles", "category", "polarity_class", "hbond_role", "green_flag", "industrial_flag"]
        print(df[cols].to_string(index=False))
        print(f"\nsolvents: {len(df)}")
        return
    if args.library_command == "export":
        path = export_library(
            args.output,
            category=args.category,
            polarity=args.polarity,
            green_only=args.green_only,
            industrial_only=args.industrial_only,
        )
        print(f"Solvent library written: {path}")
        return
    raise SystemExit("Missing solvent-library subcommand.")


def cmd_mixture(args: argparse.Namespace) -> None:
    if args.mixture_command == "generate":
        ratios = [float(item) for item in args.ratios.split(",")] if args.ratios else None
        path = generate_binary_mixtures(args.solvents, args.output, ratios=ratios, max_pairs=args.max_pairs)
        print(f"Mixture candidates written: {path}")
        return
    if args.mixture_command == "score":
        path = score_mixture_predictions(args.predictions, args.mixtures, args.output)
        print(f"Mixture scores written: {path}")
        return
    raise SystemExit("Missing mixture subcommand.")


def cmd_report(args: argparse.Namespace) -> None:
    bundle = resolve_model_or_bundle(args.model, args.bundle) if args.model or args.bundle else None
    path = write_screen_report(args.input, args.output, title=args.title, domain_csv=args.domain, bundle_dir=bundle)
    print(f"Report written: {path}")
    print(f"Report JSON written: {Path(path).with_suffix('.json')}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    bundle_path = resolve_model_or_bundle(args.model, args.bundle)
    bundle = load_bundle(bundle_path)
    manifest = pd.read_csv(bundle.root / "model_manifest.csv")
    summary = {
        "model_id": bundle.metadata.get("model_id", bundle.metadata.get("name")),
        "bundle": str(bundle.root),
        "ood_protocol": bundle.metadata.get("ood_protocol", bundle.metadata.get("benchmark", {}).get("ood_protocol")),
        "seed_count": int(manifest.shape[0]),
        "R2_mean": float(manifest["R2"].mean()),
        "R2_std": float(manifest["R2"].std(ddof=1)),
        "MAE_mean": float(manifest["MAE"].mean()),
        "RMSE_mean": float(manifest["RMSE"].mean()),
        "metadata_benchmark": bundle.metadata.get("benchmark", {}),
    }
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Benchmark written: {out}")
        return
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mica",
        description="MICA: a scientific CLI for molecular inference of compatibility and affinity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Check local runtime and Python backend dependencies.")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("models", help="List installed and available MICA model definitions.")
    models_sub = p.add_subparsers(dest="models_command", required=True)
    p_list = models_sub.add_parser("list", help="List model registry entries.")
    p_list.set_defaults(func=cmd_models)

    p = sub.add_parser("train-bundle", help="Train and export a reproducible model bundle.")
    p.add_argument("--model", help=f"Model id, e.g. {DEFAULT_MODEL_ID}.")
    p.add_argument("--learner", default="XGBoost", choices=["XGBoost", "RandomForest", "ExtraTrees", "HistGradientBoosting"])
    p.add_argument("--layer", default="CAT-Solv-47", choices=["2D-Min", "2D-Elec", "Sigma-Solv", "CAT-Solv-47"])
    p.add_argument("--output", help="Output bundle directory. Defaults to the registry path for --model.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing bundle directory.")
    p.set_defaults(func=cmd_fit_bundle)

    p = sub.add_parser("inspect", help="Inspect a MICA model bundle.")
    p.add_argument("--model", help=f"Model id. Defaults to {DEFAULT_MODEL_ID}.")
    p.add_argument("--bundle", help="Bundle directory. Defaults to the bundled top47 model.")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("validate", help="Validate an input descriptor CSV against a model bundle.")
    p.add_argument("--model", help=f"Model id. Defaults to {DEFAULT_MODEL_ID}.")
    p.add_argument("--bundle", help="Bundle directory. Defaults to the bundled top47 model.")
    p.add_argument("--input", required=True, help="Input descriptor CSV.")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("domain", help="Audit descriptor coverage and applicability-domain flags.")
    p.add_argument("--model", help=f"Model id. Defaults to {DEFAULT_MODEL_ID}.")
    p.add_argument("--bundle", help="Bundle directory. Defaults to the bundled top47 model.")
    p.add_argument("--input", required=True, help="Input descriptor CSV.")
    p.add_argument("--output", required=True, help="Output domain-audit CSV.")
    p.set_defaults(func=cmd_domain)

    p = sub.add_parser("run", help="Run high-throughput logS inference for a prepared descriptor CSV.")
    p.add_argument("--model", help=f"Model id. Defaults to {DEFAULT_MODEL_ID}.")
    p.add_argument("--bundle", help="Bundle directory. Defaults to the bundled top47 model.")
    p.add_argument("--input", required=True, help="Input feature CSV.")
    p.add_argument("--output", required=True, help="Output prediction CSV.")
    p.set_defaults(func=cmd_predict_csv)

    p = sub.add_parser("screen", help="Rank candidate solvents from a MICA prediction CSV.")
    p.add_argument("--input", required=True, help="Prediction CSV from predict-csv.")
    p.add_argument("--output", required=True, help="Output ranked CSV.")
    p.add_argument("--top", type=int, default=20, help="Top solvents per solute.")
    p.set_defaults(func=cmd_screen_csv)

    p = sub.add_parser("solvent-library", help="List or export the built-in solvent library.")
    lib_sub = p.add_subparsers(dest="library_command", required=True)
    p_list = lib_sub.add_parser("list", help="List built-in solvents.")
    p_list.add_argument("--category", help="Comma-separated category filter.")
    p_list.add_argument("--polarity", help="Comma-separated polarity_class filter.")
    p_list.add_argument("--green-only", action="store_true", help="Keep only green-flagged solvents.")
    p_list.add_argument("--industrial-only", action="store_true", help="Keep only industrial-flagged solvents.")
    p_list.set_defaults(func=cmd_solvent_library)
    p_export = lib_sub.add_parser("export", help="Export built-in solvents to CSV.")
    p_export.add_argument("--output", required=True, help="Output solvent CSV.")
    p_export.add_argument("--category", help="Comma-separated category filter.")
    p_export.add_argument("--polarity", help="Comma-separated polarity_class filter.")
    p_export.add_argument("--green-only", action="store_true", help="Keep only green-flagged solvents.")
    p_export.add_argument("--industrial-only", action="store_true", help="Keep only industrial-flagged solvents.")
    p_export.set_defaults(func=cmd_solvent_library)

    p = sub.add_parser("mixture", help="Generate or score binary solvent-mixture candidates.")
    mix_sub = p.add_subparsers(dest="mixture_command", required=True)
    p_generate = mix_sub.add_parser("generate", help="Generate binary mixture candidates from a solvent CSV.")
    p_generate.add_argument("--solvents", required=True, help="Input solvent library CSV.")
    p_generate.add_argument("--output", required=True, help="Output mixture candidate CSV.")
    p_generate.add_argument("--ratios", default="0.25,0.5,0.75", help="Comma-separated fraction_a values.")
    p_generate.add_argument("--max-pairs", type=int, help="Optional maximum number of solvent pairs.")
    p_generate.set_defaults(func=cmd_mixture)
    p_score = mix_sub.add_parser("score", help="Score mixture candidates from pure-solvent predictions.")
    p_score.add_argument("--predictions", required=True, help="Pure-solvent prediction CSV from mica run.")
    p_score.add_argument("--mixtures", required=True, help="Mixture candidate CSV from mica mixture generate.")
    p_score.add_argument("--output", required=True, help="Output mixture score CSV.")
    p_score.set_defaults(func=cmd_mixture)

    p = sub.add_parser("report", help="Generate a compact HTML report from a MICA CSV output.")
    p.add_argument("--input", required=True, help="Input CSV.")
    p.add_argument("--output", required=True, help="Output HTML report.")
    p.add_argument("--domain", help="Optional domain-audit CSV from mica domain.")
    p.add_argument("--model", help=f"Model id for model-card metadata. Defaults to {DEFAULT_MODEL_ID} if supplied without --bundle.")
    p.add_argument("--bundle", help="Bundle directory for model-card metadata.")
    p.add_argument("--title", default="MICA Screening Report", help="Report title.")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("benchmark", help="Report fixed solvent-OOD benchmark metrics for a model bundle.")
    p.add_argument("--model", help=f"Model id. Defaults to {DEFAULT_MODEL_ID}.")
    p.add_argument("--bundle", help="Bundle directory. Defaults to the bundled top47 model.")
    p.add_argument("--output", help="Optional output JSON path.")
    p.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
