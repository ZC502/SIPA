#!/usr/bin/env python3
"""
SIPA Auditor v1.1
Physical Residual Integrity (PIR) trajectory auditor

Features
--------
✔ Robust CSV loader
✔ Strict physics validator
✔ Deterministic execution
✔ Metadata logging
✔ Audit report generation
✔ CLI friendly output

Author: SIPA Research
"""

import argparse
import json
import platform
import random
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------
# Deterministic execution
# --------------------------------------------------

def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------
# Robust CSV Loader
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "frame",
    "x", "y", "z",
    "qx", "qy", "qz", "qw"
]


def load_trajectory(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        engine="c",
        low_memory=False
    )

    if len(df) == 0:
        raise ValueError("CSV file contains no rows")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("frame").reset_index(drop=True)

    return df


# --------------------------------------------------
# Physics Validator
# --------------------------------------------------

def validate_physics(df: pd.DataFrame):

    score = 1.0
    issues = []

    # Quaternion normalization
    q_norm = np.sqrt(
        df["qx"]**2 +
        df["qy"]**2 +
        df["qz"]**2 +
        df["qw"]**2
    )

    if not np.allclose(q_norm, 1.0, atol=1e-2):
        score -= 0.2
        issues.append("Quaternion normalization drift")

    # Position continuity
    pos = df[["x", "y", "z"]].values
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1)

    if np.max(vel) > 50:
        score -= 0.3
        issues.append("Unphysical velocity spike")

    # Frame continuity
    frames = df["frame"].values
    if not np.all(np.diff(frames) == 1):
        score -= 0.2
        issues.append("Frame discontinuity")

    score = max(score, 0.0)

    return score, issues


# --------------------------------------------------
# PIR Computation
# --------------------------------------------------

def compute_pir(df: pd.DataFrame, dt: float):

    pos = df[["x", "y", "z"]].values

    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt

    residual = np.linalg.norm(acc, axis=1)

    pir = np.exp(-np.mean(residual))

    return float(pir), residual


# --------------------------------------------------
# Rating
# --------------------------------------------------

def pir_rating(pir):
    
    if pir >= 0.9:
        return "A"
    elif pir >= 0.75:
        return "B"
    elif pir >= 0.6:
        return "C"
    elif pir >= 0.4:
        return "D"
    else:
        return "F" 

# --------------------------------------------------
# Report Writer
# --------------------------------------------------

def write_report(output_dir, input_file, df, pir, rating, validator_score, issues):

    report = f"""
SIPA Physical Audit Report
==========================

Input trajectory : {input_file}

Frames           : {len(df)}
Validator score  : {validator_score:.3f}

PIR Score        : {pir:.3f}
Rating           : {rating}

Validator Issues
----------------
"""

    if issues:
        for i in issues:
            report += f"- {i}\n"
    else:
        report += "None\n"

    report_path = output_dir / "audit_summary.txt"

    with open(report_path, "w") as f:
        f.write(report)

    return report_path


# --------------------------------------------------
# Metadata Logger
# --------------------------------------------------

def write_metadata(output_dir, input_file, dt, pir, validator_score):

    metadata = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_file": str(input_file),
        "dt": dt,
        "pir": pir,
        "validator_score": validator_score,
        "repo_root": str(_REPO_ROOT)
    }

    meta_path = output_dir / "audit_metadata.json"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="SIPA Physical Residual Integrity Auditor"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input trajectory CSV"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1/60,
        help="Simulation timestep"
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed"
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    set_deterministic(args.seed)

    input_csv = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[SIPA] Loading trajectory...")
    df = load_trajectory(input_csv)

    print("[SIPA] Validating physics consistency...")
    validator_score, issues = validate_physics(df)

    print("[SIPA] Computing PIR...")
    pir, residual = compute_pir(df, args.dt)

    rating = pir_rating(pir)

    print(f"[SIPA] PIR Score: {pir:.3f}")
    print(f"[SIPA] Rating: {rating}")

    report = write_report(
        output_dir,
        input_csv,
        df,
        pir,
        rating,
        validator_score,
        issues
    )

    meta = write_metadata(
        output_dir,
        input_csv,
        args.dt,
        pir,
        validator_score
    )

    print("\n[SIPA] Audit completed")
    print(f"[SIPA] Report: {report}")
    print(f"[SIPA] Metadata: {meta}\n")


if __name__ == "__main__":
    main()
