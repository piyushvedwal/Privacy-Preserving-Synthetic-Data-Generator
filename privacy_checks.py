# privacy_checks.py
# Record-level privacy checks for synthetic data: Distance to Closest Record (DCR)
# and exact-copy detection.

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _encode(real: pd.DataFrame, other: pd.DataFrame) -> tuple:
    """Encode both frames into the same numeric space.

    Numeric columns are min-max scaled using the real data's range; categorical
    columns are one-hot encoded with categories taken from both frames, so every
    column contributes on a comparable 0-1 scale.
    """
    columns = [c for c in real.columns if c in other.columns]
    real = real[columns].copy()
    other = other[columns].copy()

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(real[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]

    parts_real, parts_other = [], []

    if numeric_cols:
        r = real[numeric_cols].apply(pd.to_numeric, errors="coerce")
        o = other[numeric_cols].apply(pd.to_numeric, errors="coerce")
        lo, hi = r.min(), r.max()
        span = (hi - lo).replace(0, 1)
        r = ((r - lo) / span).fillna(0.5)
        o = ((o - lo) / span).fillna(0.5)
        parts_real.append(r.to_numpy(dtype=float))
        parts_other.append(o.to_numpy(dtype=float))

    if categorical_cols:
        combined = pd.concat(
            [real[categorical_cols].astype(str), other[categorical_cols].astype(str)],
            keys=["real", "other"],
        )
        dummies = pd.get_dummies(combined, columns=categorical_cols, dtype=float)
        parts_real.append(dummies.loc["real"].to_numpy())
        parts_other.append(dummies.loc["other"].to_numpy())

    if not parts_real:
        raise ValueError("No overlapping columns to compare.")

    return np.hstack(parts_real), np.hstack(parts_other)


def _risk_level(exact_match_rate: float, dcr_ratio: Optional[float]) -> str:
    if exact_match_rate > 0.05 or (dcr_ratio is not None and dcr_ratio < 0.5):
        return "High"
    if exact_match_rate > 0.01 or (dcr_ratio is not None and dcr_ratio < 0.9):
        return "Medium"
    return "Low"


def distance_to_closest_record(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    max_rows: int = 3000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Measure how close synthetic rows are to real rows.

    Returns DCR statistics for synthetic->real, a real->real baseline (each real
    row's distance to its nearest *other* real row), their ratio, the share of
    synthetic rows that exactly copy a real row, and an overall risk level.

    A DCR ratio well below 1 means synthetic rows sit closer to real records than
    real records sit to each other, a sign the model may be memorising data.
    """
    real = real_data.dropna(how="all")
    synth = synthetic_data.dropna(how="all")
    if len(real) < 2 or len(synth) < 1:
        raise ValueError("Need at least 2 real rows and 1 synthetic row.")

    if len(real) > max_rows:
        real = real.sample(max_rows, random_state=random_state)
    if len(synth) > max_rows:
        synth = synth.sample(max_rows, random_state=random_state)

    real_x, synth_x = _encode(real, synth)

    nn = NearestNeighbors(n_neighbors=2).fit(real_x)

    synth_dist, _ = nn.kneighbors(synth_x, n_neighbors=1)
    synth_dcr = synth_dist[:, 0]

    real_dist, _ = nn.kneighbors(real_x, n_neighbors=2)
    real_dcr = real_dist[:, 1]  # column 0 is the row itself

    exact_match_rate = float(np.mean(synth_dcr < 1e-8))
    baseline_median = float(np.median(real_dcr))
    synth_median = float(np.median(synth_dcr))
    dcr_ratio = synth_median / baseline_median if baseline_median > 0 else None

    return {
        "synthetic_dcr_median": synth_median,
        "synthetic_dcr_5th_percentile": float(np.percentile(synth_dcr, 5)),
        "real_baseline_dcr_median": baseline_median,
        "dcr_ratio": dcr_ratio,
        "exact_match_rate": exact_match_rate,
        "exact_match_count": int(np.sum(synth_dcr < 1e-8)),
        "rows_compared": {"real": len(real), "synthetic": len(synth)},
        "risk_level": _risk_level(exact_match_rate, dcr_ratio),
        "synthetic_dcr_values": synth_dcr,
        "real_dcr_values": real_dcr,
    }
