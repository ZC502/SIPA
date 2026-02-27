import numpy as np


# =========================================================
# Quality mapping
# =========================================================

def quality_level_to_score(level: str) -> float:
    """
    Map data quality label → numeric weight.
    """
    if level is None:
        return 0.7

    level = str(level).lower()

    mapping = {
        "green": 0.95,
        "yellow": 0.75,
        "red": 0.55,
    }

    return float(mapping.get(level, 0.7))


# =========================================================
# Framewise Physical Debt
# =========================================================

def compute_framewise_debt(drift: np.ndarray):
    """
    Convert drift(t) → framewise physical debt in [0,1].

    Parameters
    ----------
    drift : array-like, shape (T,)

    Returns
    -------
    debt_t : ndarray, shape (T,)
    """

    if drift is None:
        return np.zeros(1, dtype=float)

    drift = np.asarray(drift, dtype=float).reshape(-1)

    if drift.size < 2:
        return np.zeros_like(drift)

    # remove non-finite safely
    drift = np.nan_to_num(drift, nan=0.0, posinf=0.0, neginf=0.0)

    # temporal gradient = order sensitivity proxy
    grad = np.abs(np.diff(drift, prepend=drift[0]))

    # log compression (stable across scales)
    debt = np.log10(grad + 1e-12) + 6.0
    debt = np.clip(debt / 6.0, 0.0, 1.0)

    return debt.astype(float)


# =========================================================
# PIR Time Series
# =========================================================

def compute_pir_timeseries(data_quality_level: str, debt_t: np.ndarray):
    """
    Compute PIR(t) over time.

    PIR = quality_weight * (1 - normalized_debt)
    """

    debt_t = np.asarray(debt_t, dtype=float).reshape(-1)

    q_score = quality_level_to_score(data_quality_level)

    pir_t = q_score * (1.0 - debt_t)
    pir_t = np.clip(pir_t, 0.0, 1.0)

    return pir_t.astype(float)


# =========================================================
# Integrity Onset Detection
# =========================================================

def detect_integrity_onset(
    pir_t: np.ndarray,
    dt: float,
    threshold: float = 0.5,
    patience: int = 3,
):
    """
    Detect first sustained PIR drop.

    Parameters
    ----------
    pir_t : array-like
    dt : float
    threshold : float
        PIR below this is considered risk.
    patience : int
        Number of consecutive frames required.

    Returns
    -------
    dict or None
    """

    pir_t = np.asarray(pir_t, dtype=float).reshape(-1)

    if pir_t.size == 0:
        return None

    below = pir_t < threshold

    count = 0
    start_idx = None

    for i, flag in enumerate(below):
        if flag:
            if count == 0:
                start_idx = i
            count += 1

            if count >= patience:
                return {
                    "frame": int(start_idx),
                    "time_sec": float(start_idx * dt),
                    "pir_value": float(pir_t[start_idx]),
                }
        else:
            count = 0
            start_idx = None

    return None
