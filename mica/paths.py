from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PACKAGE_ROOT / "artifacts" / "mica_top47_bundle"
DEFAULT_MODEL_BUNDLE_ROOT = PACKAGE_ROOT / "artifacts" / "mica_model_bundles"
DEFAULT_TOPK_EXPERIMENT = Path(
    "/Users/catmushroomliu/Documents/code/catsolve4/experiments/topk_matched_feature_count"
)
DEFAULT_PHYSCHEM_EXPERIMENT = Path(
    "/Users/catmushroomliu/Documents/code/catsolve4/experiments/physchem_cross_20seed"
)
DEFAULT_LEARNER_ROBUSTNESS_EXPERIMENT = Path(
    "/Users/catmushroomliu/Documents/code/catsolve4/experiments/learner_robustness_20seed"
)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
