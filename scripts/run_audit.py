#!/usr/bin/env python3
"""
SIPA: Spatial Intelligence Physical Audit
----------------------------------------

A trajectory-level physical consistency diagnostic tool based on
the Non-Associative Residual Hypothesis (NARH).

License:
- Research use permitted with attribution.
- Commercial use requires a separate license agreement.

Patent status: pending.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

# ============================================================
# Phase 0 — Bootstrap (robust repo-local imports)
# ============================================================

def _bootstrap_repo() -> Path:
    """
    Ensure the repository root and local extensions are discoverable
    when executing as a standalone script.
    """
    this_file = Path(__file__).resolve()

    # Locate repo root (folder containing "scripts")
    repo_root = None
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "scripts").exists():
            repo_root = parent
            break

    if repo_root is None:
        repo_root = this_file.parent

    # Add repo root to sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Add octonion extension if present
    ext_path = repo_root / "exts" / "octonion_time"
    if ext_path.exists() and str(ext_path) not in sys.path:
        sys.path.insert(0, str(ext_path))

    return repo_root


_REPO_ROOT = _bootstrap_repo()


# ============================================================
# Phase 1 — Core Imports
# ============================================================

try:
    from scripts.sipa_video_auditor import run_residual_audit
    from scripts.calculate_debt import compute_debt_and_pir
    from scripts.audit_visualization import plot_pir_evolution
except ImportError as e:
    print(f"[SIPA][FATAL] Module import failed: {e}", file=sys.stderr)
    sys.exit(1)


# ============================================================
# Phase 2 — Lightweight Validator
# ============================================================

def validate_csv_sanity(csv_path: Path) -> float:
    """
    Compute a conservative data integrity score in [0,1].

    Checks:
    - Required columns
    - NaNs
    - Minimal length
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        required = {"x", "y", "z", "qx", "qy", "qz", "qw"}
        if not required.issubset(df.columns):
            return 0.1

        if df.isnull().values.any():
            return 0.4

        if len(df) < 10:
            return 0.3

        return 0.95

    except Exception:
        return 0.0


# ============================================================
# Phase 3 — Orchestration Engine
# ============================================================

def execute_pipeline(
    input_csv: Path,
    output_dir: Path,
    dt: float,
    validator_override: Optional[float],
    verbose: bool,
    branding: bool,
):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if verbose:
            print(msg)

    # Header
    if branding:
        print("\n" + "=" * 60)
        print("SIPA — Spatial Intelligence Physical Audit")
        print("Non-Associative Residual Hypothesis (NARH) Evaluation")
        print("=" * 60)
    else:
        log("[SIPA] Starting audit...")

    # --------------------------------------------------------
    # 1. Data Validation
    # --------------------------------------------------------
    score = (
        validator_override
        if validator_override is not None
        else validate_csv_sanity(input_csv)
    )

    log(f"[SIPA] Data integrity score: {score:.2f}")

    # --------------------------------------------------------
    # 2. Residual Audit
    # --------------------------------------------------------
    log("[SIPA] Running residual audit...")
    residual_summary = run_residual_audit(input_csv)

    # --------------------------------------------------------
    # 3. PIR Computation
    # --------------------------------------------------------
    log("[SIPA] Computing Physical Integrity Rating (PIR)...")

    pir_t, debt_t, onset_info = compute_debt_and_pir(
        csv_path=input_csv,
        dt=dt,
        residual_summary=residual_summary,
        data_quality=score,
    )

    # --------------------------------------------------------
    # 4. Visualization
    # --------------------------------------------------------
    log("[SIPA] Generating diagnostic visualization...")

    plot_pir_evolution(
        pir_t=pir_t,
        debt_t=debt_t,
        dt=dt,
        onset_info=onset_info,
        save_dir=output_dir,
        validator_score=score,
    )

    # --------------------------------------------------------
    # 5. Final Verdict
    # --------------------------------------------------------
    final_pir = float(pir_t[-1])

    if final_pir >= 0.85:
        grade = "A"
    elif final_pir >= 0.70:
        grade = "B"
    elif final_pir >= 0.50:
        grade = "C"
    elif final_pir >= 0.30:
        grade = "D"
    else:
        grade = "F"

    print("\n[SIPA] Final PIR:", f"{final_pir:.4f}")
    print("[SIPA] Rating:", grade)
    print("[SIPA] Output directory:", output_dir.resolve())

    if branding:
        print("=" * 60 + "\n")


# ============================================================
# Phase 4 — CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIPA — Spatial Intelligence Physical Audit"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to SIPA-compatible CSV trajectory",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs/)",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Timestep in seconds (if no timestamp column)",
    )

    parser.add_argument(
        "--validator-score",
        type=float,
        default=None,
        help="Optional override for data integrity score [0–1]",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )

    parser.add_argument(
        "--branding",
        action="store_true",
        help="Enable branded header output",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        execute_pipeline(
            input_csv=args.input,
            output_dir=args.out,
            dt=args.dt,
            validator_override=args.validator_score,
            verbose=args.verbose,
            branding=args.branding,
        )
    except Exception as e:
        print(f"[SIPA][ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
