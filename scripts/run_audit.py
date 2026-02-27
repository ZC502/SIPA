# ============================================================
# SIPA v1.0 — Unified Audit Entry
# ------------------------------------------------------------
# Usage:
# python scripts/run_audit.py \
#     --input data/traj.csv \
#     --format csv \
#     --dt 0.02
# ============================================================

import argparse
import os
import time
import numpy as np
import pandas as pd

from sipa_video_auditor import SIPAVideoAuditor


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
    if arr.shape[1] != 7:
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

    # --------------------------------------------------------
    # run auditor
    # --------------------------------------------------------
    auditor = SIPAVideoAuditor(dt=args.dt)

    print("[SIPA] Running audit...")
    report = auditor.analyze(poses, save_dir=run_dir)

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------
    print("\n=== SIPA SUMMARY ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print(f"\n[SIPA] Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
