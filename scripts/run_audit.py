# ============================================================
# SIPA v1.0 — Unified Audit Entry (Hardened)
# ============================================================

import argparse
import os
import time
import numpy as np
import pandas as pd

from sipa_video_auditor import SIPAVideoAuditor


# ============================================================
# Input Validator
# ============================================================

class SIPAInputValidator:
    """
    Lightweight input quality gate for SIPA.
    Produces a traffic-light quality label.
    """

    def __init__(self):
        self.messages = []
        self.score_penalty = 0.0

    # -----------------------------------------------------
    # Main entry
    # -----------------------------------------------------
    def validate(self, poses: np.ndarray, dt: float):
        """
        poses: (T,7) -> [x,y,z,qx,qy,qz,qw]
        dt: expected timestep
        """
        self.messages.clear()
        self.score_penalty = 0.0

        # 🔒 shape guard (NEW)
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

    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------
    def _check_nan_inf(self, poses):
        if not np.isfinite(poses).all():
            self.messages.append(
                "[RED] NaN or Inf detected in input trajectory."
            )
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

    # -----------------------------------------------------
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

    poses = df[required].values.astype(float)
    return poses


def load_npy(path):
    arr = np.load(path)

    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"NPY must have shape (T,7), got {arr.shape}"
        )

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

    # --------------------------------------------------------
    # timestamped output
    # --------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"audit_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print("[SIPA] Loading trajectory...")
    poses = load_auto(args.input, args.format)
    print(f"[SIPA] Loaded {len(poses)} frames")

    # ========================================================
    # 🔥 NEW: INPUT QUALITY GATE
    # ========================================================
    validator = SIPAInputValidator()
    quality_level, messages = validator.validate(poses, args.dt)

    print(f"\n[SIPA] Input Quality: {quality_level}")
    for m in messages:
        print(m)

    if quality_level == "RED":
        print(
            "[WARNING] Input data quality is poor. "
            "Physical conclusions may be unreliable."
        )

    # --------------------------------------------------------
    # run auditor
    # --------------------------------------------------------
    auditor = SIPAVideoAuditor(dt=args.dt)

    print("\n[SIPA] Running audit...")
    report = auditor.analyze(poses, save_dir=run_dir)

    # 🔥 attach quality to report
    report["input_quality"] = quality_level

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------
    print("\n=== SIPA SUMMARY ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print(f"\n[SIPA] Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
