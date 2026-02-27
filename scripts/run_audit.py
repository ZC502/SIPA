# ============================================================
# SIPA v1.0 — Unified Audit Entry (Hardened)
# ============================================================

import argparse
import os
import time
import json
import numpy as np
import pandas as pd

from sipa_video_auditor import SIPAVideoAuditor
from audit_visualization import plot_pir_evolution


# ============================================================
# PIR utilities
# ============================================================

def normalize_physical_debt(associator_mag: np.ndarray):
    if associator_mag is None or len(associator_mag) == 0:
        return 0.0

    mean_val = float(np.mean(associator_mag))
    score = np.log10(mean_val + 1e-12) + 6.0
    score = np.clip(score / 6.0, 0.0, 1.0)
    return float(score)


def quality_level_to_score(level: str) -> float:
    mapping = {
        "GREEN": 0.95,
        "YELLOW": 0.75,
        "RED": 0.40,
    }
    return mapping.get(level.upper(), 0.60)


def compute_pir(data_quality_level: str, physical_debt_norm: float):
    q_score = quality_level_to_score(data_quality_level)

    pir = q_score * (1.0 - physical_debt_norm)
    pir = float(np.clip(pir, 0.0, 1.0))

    if pir >= 0.85:
        grade, label = "A", "High Integrity"
    elif pir >= 0.70:
        grade, label = "B", "Acceptable"
    elif pir >= 0.50:
        grade, label = "C", "Speculative"
    elif pir >= 0.30:
        grade, label = "D", "High Risk"
    else:
        grade, label = "F", "Critical"

    return pir, grade, label


# ============================================================
# Framewise utilities 
# ============================================================

def compute_framewise_debt(associator_series: np.ndarray):
    """Map associator magnitude to framewise normalized debt."""
    associator_series = np.asarray(associator_series, dtype=float)

    if associator_series.size == 0:
        return np.zeros(1)

    log_scaled = np.log10(associator_series + 1e-12) + 6.0
    debt = np.clip(log_scaled / 6.0, 0.0, 1.0)
    return debt


def compute_pir_timeseries(quality_level: str, debt_t: np.ndarray):
    q_score = quality_level_to_score(quality_level)
    pir_t = q_score * (1.0 - debt_t)
    return np.clip(pir_t, 0.0, 1.0)


def detect_integrity_onset(pir_t: np.ndarray, dt: float, threshold=0.5):
    """Detect first time PIR drops below threshold."""
    below = np.where(pir_t < threshold)[0]
    if len(below) == 0:
        return None

    idx = int(below[0])
    return {
        "frame_index": idx,
        "time_sec": float(idx * dt),
        "threshold": float(threshold),
    }


# ============================================================
# Input Validator
# ============================================================

class SIPAInputValidator:
    def __init__(self):
        self.messages = []
        self.score_penalty = 0.0

    def validate(self, poses: np.ndarray, dt: float):
        self.messages.clear()
        self.score_penalty = 0.0

        if poses.ndim != 2 or poses.shape[1] != 7:
            self.messages.append(
                f"[RED] Invalid pose shape {poses.shape}, expected (T,7)."
            )
            return "RED", list(self.messages)

        self._check_nan_inf(poses)
        self._check_quaternion_norm(poses)
        self._check_temporal_jitter(poses, dt)
        self._check_position_jumps(poses)

        level = self._final_grade()
        return level, list(self.messages)

    def _check_nan_inf(self, poses):
        if not np.isfinite(poses).all():
            self.messages.append("[RED] NaN or Inf detected in input trajectory.")
            self.score_penalty += 3.0

    def _check_quaternion_norm(self, poses):
        q = poses[:, 3:7]
        norms = np.linalg.norm(q, axis=1)
        deviation = np.abs(norms - 1.0)

        max_dev = float(deviation.max())

        if max_dev > 5e-2:
            self.messages.append(
                f"[RED] Quaternion severely unnormalized (max dev={max_dev:.3e})."
            )
            self.score_penalty += 3.0
        elif max_dev > 1e-3:
            self.messages.append(
                f"[YELLOW] Quaternion slightly off unit norm (max dev={max_dev:.3e})."
            )
            self.score_penalty += 1.0

    def _check_temporal_jitter(self, poses, dt):
        pos = poses[:, :3]
        vel = np.diff(pos, axis=0) / max(dt, 1e-8)

        if len(vel) < 3:
            return

        speed = np.linalg.norm(vel, axis=1)
        jitter = np.std(speed) / (np.mean(speed) + 1e-8)

        if jitter > 1.0:
            self.messages.append(
                f"[YELLOW] High temporal jitter detected (ratio={jitter:.2f})."
            )
            self.score_penalty += 1.0

    def _check_position_jumps(self, poses):
        pos = poses[:, :3]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)

        if len(step) == 0:
            return

        median = np.median(step) + 1e-8
        max_jump = float(step.max() / median)

        if max_jump > 50:
            self.messages.append(
                f"[RED] Extreme position jump detected (x{max_jump:.1f})."
            )
            self.score_penalty += 3.0
        elif max_jump > 10:
            self.messages.append(
                f"[YELLOW] Large position jump detected (x{max_jump:.1f})."
            )
            self.score_penalty += 1.0

    def _final_grade(self):
        if self.score_penalty >= 3.0:
            return "RED"
        elif self.score_penalty >= 1.0:
            return "YELLOW"
        return "GREEN"


# ============================================================
# IO helpers
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    required = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[required].values.astype(float)


def load_npy(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"NPY must have shape (T,7), got {arr.shape}")
    return arr.astype(float)


def load_auto(path, fmt):
    if fmt == "csv":
        return load_csv(path)
    elif fmt == "npy":
        return load_npy(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SIPA trajectory physical consistency audit"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", required=True, choices=["csv", "npy"])
    parser.add_argument("--dt", required=True, type=float)
    parser.add_argument("--outdir", default="outputs")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"audit_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print("[SIPA] Loading trajectory...")
    poses = load_auto(args.input, args.format)
    print(f"[SIPA] Loaded {len(poses)} frames")

    # ---------- validation ----------
    validator = SIPAInputValidator()
    quality_level, messages = validator.validate(poses, args.dt)

    print(f"\n[SIPA] Input Quality: {quality_level}")
    for m in messages:
        print(m)

    # ---------- audit ----------
    auditor = SIPAVideoAuditor(dt=args.dt)
    print("\n[SIPA] Running audit...")
    report = auditor.analyze(poses, save_dir=run_dir)

    assoc_series = np.asarray(
        report.get("associator_magnitude_series", []), dtype=float
    )

    debt_norm = normalize_physical_debt(assoc_series)
    pir, grade, label = compute_pir(quality_level, debt_norm)

    # ---------- time series ----------
    debt_t = compute_framewise_debt(assoc_series)
    pir_t = compute_pir_timeseries(quality_level, debt_t)
    onset = detect_integrity_onset(pir_t, args.dt)

    # ---------- plot ----------
    png_path, pdf_path, _ = plot_pir_evolution(
        pir_t,
        debt_t,
        args.dt,
        onset,
        save_dir=run_dir,
    )

    # ---------- report ----------
    report.update(
        {
            "input_quality": quality_level,
            "physical_debt_norm": debt_norm,
            "pir_score": pir,
            "pir_grade": grade,
            "pir_label": label,
            "integrity_onset": onset,
        }
    )

    json_path = os.path.join(run_dir, "sipa_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---------- terminal ----------
    quality_score = quality_level_to_score(quality_level)

    print("\n=== SIPA SUMMARY ===")
    print(f"[SIPA] FINAL RATING: {grade} ({label})")
    print(f"• Data Quality: {quality_score*100:.1f}%")
    print(f"• Physical Consistency: {(1-debt_norm)*100:.1f}%")
    print(f"\n[SIPA] Integrity onset:", onset)
    print(f"[SIPA] Figure saved:", png_path)
    print(f"[SIPA] Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
