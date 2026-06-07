from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "outputs"
BUNDLE = ROOT / "artifacts" / "mica_top47_bundle"
TOPK_EXPERIMENT = Path("/Users/catmushroomliu/Documents/code/catsolve4/experiments/topk_matched_feature_count")
PHYSCHEM_EXPERIMENT = Path("/Users/catmushroomliu/Documents/code/catsolve4/experiments/physchem_cross_20seed")


def run_command(cmd: list[str]) -> tuple[float, int, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    return elapsed, proc.returncode, proc.stdout, proc.stderr


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_benchmark_inputs() -> dict[str, Path]:
    import sys

    sys.path.insert(0, str(PHYSCHEM_EXPERIMENT))
    from run_physchem_cross_20seed import build_model_frames, build_test_manifest

    OUT.mkdir(parents=True, exist_ok=True)
    frames, _ = build_model_frames()
    df = frames["v2_physchem_class_cross"]
    _, _, test_ids = build_test_manifest(frames["v1"])
    ood = df[df["Common_RowID"].isin(test_ids)].copy().reset_index(drop=True)

    paths = {}
    for name, n_repeat in [("ood_5424", 1), ("ood_10848", 2), ("ood_54240", 10)]:
        repeated = pd.concat([ood] * n_repeat, ignore_index=True)
        repeated["Benchmark_RowID"] = np.arange(len(repeated))
        path = OUT / f"{name}_full98_features.csv"
        repeated.to_csv(path, index=False)
        paths[name] = path
    return paths


def evaluate_prediction_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "LogS" not in df.columns:
        return {}
    return metric_dict(df["LogS"].to_numpy(), df["pred_logS_mean"].to_numpy())


def compute_bundle_check_metrics() -> dict[str, float]:
    check = pd.read_csv(BUNDLE / "ood_predictions_bundle_check.csv")
    rows = []
    for seed, group in check.groupby("seed"):
        row = {"seed": int(seed), **metric_dict(group["y_true"].to_numpy(), group["y_pred"].to_numpy())}
        rows.append(row)
    df = pd.DataFrame(rows)
    return {
        "seed_count": int(len(df)),
        "R2_mean": float(df["R2"].mean()),
        "R2_std": float(df["R2"].std(ddof=1)),
        "MAE_mean": float(df["MAE"].mean()),
        "RMSE_mean": float(df["RMSE"].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = build_benchmark_inputs()

    summary: dict[str, object] = {
        "tool": "MICA",
        "bundle": str(BUNDLE),
        "accuracy_from_bundle_check": compute_bundle_check_metrics(),
        "runtime": [],
    }

    quick_commands = [
        ("doctor", ["mica", "doctor"]),
        ("inspect", ["mica", "inspect"]),
        ("validate_5424", ["mica", "validate", "--input", str(inputs["ood_5424"])]),
    ]
    for name, cmd in quick_commands:
        elapsed, code, stdout, stderr = run_command(cmd)
        summary["runtime"].append(
            {
                "task": name,
                "rows": 0 if "5424" not in name else 5424,
                "seconds": elapsed,
                "exit_code": code,
                "stdout_tail": stdout.strip().splitlines()[-5:],
                "stderr_tail": stderr.strip().splitlines()[-5:],
            }
        )

    for name, input_path in inputs.items():
        n_rows = int(pd.read_csv(input_path, usecols=["Benchmark_RowID"]).shape[0])
        pred_path = OUT / f"{name}_predictions.csv"
        screen_path = OUT / f"{name}_screen_top10.csv"

        elapsed, code, stdout, stderr = run_command(
            ["mica", "run", "--input", str(input_path), "--output", str(pred_path)]
        )
        metrics = evaluate_prediction_file(pred_path) if code == 0 else {}
        summary["runtime"].append(
            {
                "task": f"run_{name}",
                "rows": n_rows,
                "seconds": elapsed,
                "rows_per_second": n_rows / elapsed if elapsed > 0 else None,
                "exit_code": code,
                **metrics,
                "stdout_tail": stdout.strip().splitlines()[-5:],
                "stderr_tail": stderr.strip().splitlines()[-5:],
            }
        )

        elapsed, code, stdout, stderr = run_command(
            ["mica", "screen", "--input", str(pred_path), "--output", str(screen_path), "--top", "10"]
        )
        out_rows = int(pd.read_csv(screen_path).shape[0]) if code == 0 else 0
        summary["runtime"].append(
            {
                "task": f"screen_{name}",
                "rows": n_rows,
                "output_rows": out_rows,
                "seconds": elapsed,
                "rows_per_second": n_rows / elapsed if elapsed > 0 else None,
                "exit_code": code,
                "stdout_tail": stdout.strip().splitlines()[-5:],
                "stderr_tail": stderr.strip().splitlines()[-5:],
            }
        )

    (OUT / "mica_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    runtime_df = pd.DataFrame(summary["runtime"])
    runtime_df.to_csv(OUT / "mica_benchmark_runtime.csv", index=False)

    report = [
        "# MICA CLI Benchmark",
        "",
        "## Accuracy",
        "",
        pd.DataFrame([summary["accuracy_from_bundle_check"]]).to_markdown(index=False),
        "",
        "## Runtime",
        "",
        runtime_df[
            [
                "task",
                "rows",
                "seconds",
                "rows_per_second",
                "exit_code",
                "R2",
                "MAE",
                "RMSE",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (OUT / "mica_benchmark_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "mica_benchmark_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
