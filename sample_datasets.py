# sample_datasets.py
# Built-in demo datasets (bundled with scikit-learn, no download needed).

from typing import Callable, Dict

import pandas as pd
from sklearn import datasets


def _labelled(loader: Callable, target_name: str) -> pd.DataFrame:
    bunch = loader(as_frame=True)
    df = bunch.frame.copy()
    df[target_name] = pd.Categorical.from_codes(bunch.target, bunch.target_names).astype(str)
    if "target" in df.columns and target_name != "target":
        df = df.drop(columns="target")
    return df


def _diabetes() -> pd.DataFrame:
    return datasets.load_diabetes(as_frame=True, scaled=False).frame


SAMPLE_DATASETS: Dict[str, Dict] = {
    "Iris (150 rows, flower measurements)": {
        "loader": lambda: _labelled(datasets.load_iris, "species"),
        "target": "species",
    },
    "Wine (178 rows, chemical analysis)": {
        "loader": lambda: _labelled(datasets.load_wine, "cultivar"),
        "target": "cultivar",
    },
    "Breast Cancer (569 rows, medical diagnostics)": {
        "loader": lambda: _labelled(datasets.load_breast_cancer, "diagnosis"),
        "target": "diagnosis",
    },
    "Diabetes (442 rows, patient records)": {
        "loader": _diabetes,
        "target": None,
    },
}


def load_sample(name: str) -> pd.DataFrame:
    """Load one of the built-in sample datasets by display name."""
    if name not in SAMPLE_DATASETS:
        raise KeyError(f"Unknown sample dataset: {name}")
    df = SAMPLE_DATASETS[name]["loader"]()
    df.columns = [str(c).replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    return df
