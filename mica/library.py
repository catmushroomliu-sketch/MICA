from __future__ import annotations

from pathlib import Path

import pandas as pd


LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "solvent_library.csv"


def load_solvent_library() -> pd.DataFrame:
    return pd.read_csv(LIBRARY_PATH)


def filter_library(
    category: str | None = None,
    polarity: str | None = None,
    green_only: bool = False,
    industrial_only: bool = False,
) -> pd.DataFrame:
    df = load_solvent_library()
    if category:
        allowed = {item.strip() for item in category.split(",") if item.strip()}
        df = df[df["category"].isin(allowed)]
    if polarity:
        allowed = {item.strip() for item in polarity.split(",") if item.strip()}
        df = df[df["polarity_class"].isin(allowed)]
    if green_only:
        df = df[df["green_flag"].astype(str).str.lower() == "yes"]
    if industrial_only:
        df = df[df["industrial_flag"].astype(str).str.lower() == "yes"]
    return df.reset_index(drop=True)


def export_library(output: str | Path, **filters: object) -> Path:
    out = Path(output).expanduser().resolve()
    df = filter_library(**filters)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

