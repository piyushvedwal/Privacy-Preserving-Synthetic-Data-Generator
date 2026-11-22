# privacy_checks.py
# Record-level privacy checks for synthetic data: Distance to Closest Record (DCR)
# and exact-copy detection.

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def encode_frames(reference: pd.DataFrame, *others: pd.DataFrame) -> tuple:
    """Encode several frames into one shared numeric space.

    Numeric columns are min-max scaled using the reference frame's range;
    categorical columns are one-hot encoded with categories taken from all frames,
    so every column contributes on a comparable 0-1 scale. Only columns present in
    every frame are used. Returns one array per input frame.
    """
    frames = [reference, *others]
    columns = [c for c in reference.columns if all(c in f.columns for f in others)]
    if not columns:
        raise ValueError("No overlapping columns to compare.")
    frames = [f[columns].copy() for f in frames]

    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(reference[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]

    parts = [[] for _ in frames]

    if numeric_cols:
        ref = frames[0][numeric_cols].apply(pd.to_numeric, errors="coerce")
        lo, hi = ref.min(), ref.max()
        span = (hi - lo).replace(0, 1)
        for i, f in enumerate(frames):
            num = f[numeric_cols].apply(pd.to_numeric, errors="coerce")
            parts[i].append(((num - lo) / span).fillna(0.5).to_numpy(dtype=float))

    if categorical_cols:
        keys = [str(i) for i in range(len(frames))]
        combined = pd.concat([f[categorical_cols].astype(str) for f in frames], keys=keys)
        dummies = pd.get_dummies(combined, columns=categorical_cols, dtype=float)
        for i, key in enumerate(keys):
            parts[i].append(dummies.loc[key].to_numpy())

    return tuple(np.hstack(p) for p in parts)


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

    real_x, synth_x = encode_frames(real, synth)

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


def distance_membership_inference(
    train_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    max_rows: int = 3000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Distance-based membership inference attack.

    The attacker guesses that a real record was in the generator's training set if
    a synthetic record lies close to it. Members are rows the generator was trained
    on; non-members are held-out rows it never saw. An equal number of each is
    scored by the negative distance to the closest synthetic row.

    AUC 0.5 means the attacker cannot tell members from non-members (no measurable
    leakage); values towards 1.0 mean synthetic data reveals who was in training.
    """
    from sklearn.metrics import roc_auc_score

    n = min(len(train_data), len(holdout_data), max_rows)
    if n < 5:
        raise ValueError("Need at least 5 training and 5 held-out rows.")
    members = train_data.sample(n, random_state=random_state)
    non_members = holdout_data.sample(n, random_state=random_state)
    synth = synthetic_data.dropna(how="all")
    if len(synth) > max_rows:
        synth = synth.sample(max_rows, random_state=random_state)

    reference = pd.concat([train_data, holdout_data])
    _, member_x, non_member_x, synth_x = encode_frames(reference, members, non_members, synth)
    nn = NearestNeighbors(n_neighbors=1).fit(synth_x)
    d_member = nn.kneighbors(member_x)[0][:, 0]
    d_non_member = nn.kneighbors(non_member_x)[0][:, 0]

    labels = np.r_[np.ones(n), np.zeros(n)]
    scores = -np.r_[d_member, d_non_member]
    auc = float(roc_auc_score(labels, scores))
    threshold = np.median(scores)
    accuracy = float(np.mean((scores > threshold) == labels.astype(bool)))

    return {
        "membership_inference_auc": auc,
        "membership_inference_accuracy": accuracy,
        "membership_inference_advantage": max(0.0, 2 * auc - 1),
        "member_dcr_median": float(np.median(d_member)),
        "non_member_dcr_median": float(np.median(d_non_member)),
        "records_per_group": int(n),
    }
