# ============================================================
# SIPA path bootstrap (robust repo-local imports)
# ============================================================

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]

# Add repo root for absolute imports
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add octonion extension if present
_EXT_PATH = _REPO_ROOT / "exts" / "octonion_time"
if _EXT_PATH.exists() and str(_EXT_PATH) not in sys.path:
    sys.path.insert(0, str(_EXT_PATH))

#!/usr/bin/env python3
"""
SIPA One-Click Audit CLI (Production)

Pipeline:
    CSV → Validator → Residual Auditor → Debt/PIR → Visualization → Verdict

This script is intentionally thin and acts as the orchestration layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# --- local modules ---
from scripts.sipa_video_auditor import run_residual_audit
from scripts.calculate_debt import compute_debt_and_pir
from scripts.audit_visualization import plot_pir_evolution


# ============================================================
# Optional lightweight validator
# (kept here to avoid hard dependency explosion)
# ============================================================

def validate_csv_sanity(csv_path: Path) -> float:
    """
    Returns a data quality score in [0, 1].

    This is intentionally conservative and cheap.
    You can later swap in a learned validator.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if len(df) < 10:
        return 0.3

    nan_ratio = df.isna().mean().mean()

    if nan_ratio > 0.2:
        return 0.4
    if nan_ratio > 0.05:
        return 0.7

    return 0.95


# ============================================================
# Core runner
# ============================================================

def run_audit(
    input_csv: Path,
    output_dir: Path,
    dt: float,
    validator_score: Optional[float],
):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[SIPA] =========================================")
    print("[SIPA] Starting Physical Integrity Audit")
    print("[SIPA] =========================================")

    # --------------------------------------------------------
    # 1. Validator
    # --------------------------------------------------------
    if validator_score is None:
        print("[SIPA] Running data validator...")
        validator_score = validate_csv_sanity(input_csv)

    print(f"[SIPA] Data Quality Score: {validator_score:.2f}")

    # --------------------------------------------------------
    # 2. Residual auditor
    # --------------------------------------------------------
    print("[SIPA] Running residual auditor...")
    residual_summary = run_residual_audit(input_csv)

    # --------------------------------------------------------
    # 3. Debt + PIR
    # --------------------------------------------------------
    print("[SIPA] Computing physical debt and PIR...")
    pir_t, debt_t, onset_info = compute_debt_and_pir(
        csv_path=input_csv,
        dt=dt,
        residual_summary=residual_summary,
        data_quality=validator_score,
    )

    # --------------------------------------------------------
    # 4. Visualization (the “verdict sheet”)
    # --------------------------------------------------------
    print("[SIPA] Rendering diagnostic figure...")
    plot_pir_evolution(
        pir_t=pir_t,
        debt_t=debt_t,
        dt=dt,
        onset_info=onset_info,
        save_dir=output_dir,
        validator_score=validator_score,
    )

    # --------------------------------------------------------
    # 5. Terminal verdict
    # --------------------------------------------------------
    final_pir = float(pir_t[-1])

    if final_pir > 0.8:
        grade = "A (PHYSICALLY CONSISTENT)"
    elif final_pir > 0.6:
        grade = "B (STABLE)"
    elif final_pir > 0.4:
        grade = "C (SPECULATIVE)"
    else:
        grade = "D (PHYSICALLY UNSTABLE)"

    print("\n[SIPA] FINAL RATING:", grade)
    print(f"[SIPA] Final PIR: {final_pir:.3f}")
    print(f"[SIPA] Outputs saved to: {output_dir}")
    print("[SIPA] =========================================\n")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIPA Physical Integrity Auditor"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input SIPA CSV log",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation timestep (seconds)",
    )

    parser.add_argument(
        "--validator-score",
        type=float,
        default=None,
        help="Optional override for data quality score (0–1)",
    )

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        run_audit(
            input_csv=args.input,
            output_dir=args.out,
            dt=args.dt,
            validator_score=args.validator_score,
        )

    except Exception as e:
        print(f"[SIPA][ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
