#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/sipa_yumi_audit.py

Minimal offline CSV entrypoint for SIPA YuMi.

Responsibilities
----------------
- Load a YuMi-compatible CSV file
- Auto-detect / normalize joint columns and time column
- Convert units to radians
- Build JointSample objects
- Call SIPAYuMiEngine.analyze_batch(...)
- Print a concise console report

Not included yet
----------------
- Full ABB-specific CSV dialect coverage
- Plot generation
- Report file export
- Real YuMi nominal FK injection from kinematics/yumi.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.sipa_yumi_engine import (
    SIPAYuMiEngine,
    YumiEngineConfig,
    build_joint_samples,
    convert_joint_matrix_to_rad,
    detect_unit_from_joint_matrix,
)


# ============================================================
# CSV loader
# ============================================================

def load_yumi_csv(path: str, dt_fallback: float = 0.10) -> tuple[np.ndarray, np.ndarray, str, pd.DataFrame]:
    """
    Minimal YuMi CSV loader.

    Supported patterns (v1)
    -----------------------
    1) Headered CSV with J1..J7 and optional TIME
    2) Headerless numeric CSV where first 7 columns are joints

    Returns
    -------
    q_matrix : (N, 7)
    timestamps : (N,)
    detected_source_format : str
    raw_df : DataFrame
    """
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:
        df = pd.read_csv(path, comment="#", encoding="latin1")

    joint_cols = [f"J{i}" for i in range(1, 8)]
    has_named_joints = all(c in df.columns for c in joint_cols)

    if has_named_joints:
        source_format = "headered_joint_csv"
        q_df = df[joint_cols].apply(pd.to_numeric, errors="coerce")
        q_df = q_df.dropna()
        df = df.loc[q_df.index].copy()

        if "TIME" in df.columns:
            timestamps = pd.to_numeric(df["TIME"], errors="coerce").values
        else:
            timestamps = np.arange(len(q_df), dtype=float) * float(dt_fallback)

        q_matrix = q_df.values
        return q_matrix, timestamps, source_format, df

    # fallback: headerless numeric CSV
    df2 = pd.read_csv(path, header=None)
    if df2.shape[1] < 7:
        raise ValueError("CSV must contain at least 7 joint columns")

    q_df = df2.iloc[:, :7].apply(pd.to_numeric, errors="coerce").dropna()
    df2 = df2.loc[q_df.index].copy()

    q_matrix = q_df.values
    timestamps = np.arange(len(q_matrix), dtype=float) * float(dt_fallback)
    source_format = "headerless_numeric_csv"
    return q_matrix, timestamps, source_format, df2


# ============================================================
# CLI reporting
# ============================================================

def print_batch_report(report) -> None:
    print("\n" + "=" * 60)
    print("SIPA for ABB YuMi - Offline Audit")
    print("=" * 60)

    print(f"Total samples: {report.total_samples}")
    print(f"Associator events: {report.summary.get('associator_event_count', 0)}")
    print(f"Alarm count: {report.summary.get('alarm_count', 0)}")

    if "associator_raw_peak" in report.summary:
        print(f"Associator raw peak: {report.summary['associator_raw_peak']:.6f}")
    if "associator_norm_peak" in report.summary:
        print(f"Associator normalized peak: {report.summary['associator_norm_peak']:.6f}")
    if "tcp_step_mm_peak" in report.summary:
        print(f"Estimated TCP step peak: {report.summary['tcp_step_mm_peak']:.3f} mm")

    if report.associator_events:
        print("\nTop associator events:")
        for ev in report.associator_events[:10]:
            tcp_info = f", tcp_step={ev.tcp_step_mm:.3f} mm" if ev.tcp_step_mm is not None and np.isfinite(ev.tcp_step_mm) else ""
            print(
                f"- [{ev.severity}] frame={ev.frame_idx}, t={ev.timestamp:.3f}s, "
                f"score={ev.normalized_score:.4f}, raw={ev.raw_score:.4f}{tcp_info}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to YuMi joint CSV")
    parser.add_argument("--dt", type=float, default=0.10, help="Fallback dt if CSV has no TIME column")
    parser.add_argument("--unit", choices=["auto", "deg", "rad"], default="auto")
    parser.add_argument("--disable-fk", action="store_true", help="Disable nominal FK / TCP estimation")
    args = parser.parse_args()

    q_matrix, timestamps, source_format, raw_df = load_yumi_csv(args.input, dt_fallback=args.dt)

    unit = args.unit
    if unit == "auto":
        unit = detect_unit_from_joint_matrix(q_matrix)

    q_matrix = convert_joint_matrix_to_rad(q_matrix, unit)

    samples = build_joint_samples(
        q_matrix=q_matrix,
        timestamps=timestamps,
        source="csv",
        meta_list=[{"source_format": source_format} for _ in range(len(q_matrix))],
    )

    cfg = YumiEngineConfig(enable_nominal_fk=not args.disable_fk)
    engine = SIPAYuMiEngine(config=cfg)
    report = engine.analyze_batch(samples)

    print(f"[SIPA] Source format: {source_format}")
    print(f"[SIPA] Auto/selected unit: {unit}")
    print(f"[SIPA] Samples: {len(samples)}")
    print(f"[SIPA] Nominal FK enabled: {not args.disable_fk}")

    print_batch_report(report)


if __name__ == "__main__":
    main()
