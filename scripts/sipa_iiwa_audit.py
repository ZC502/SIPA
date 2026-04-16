#!/usr/bin/env python3

import sys
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

"""
SIPA v2.3 Industrial Solver Audit Edition
LBR iiwa Edition

Default Robot
-------------
KUKA LBR iiwa 14 R820

Optional
--------
iiwa7

Features
--------
Forward Kinematics
TCP Jump Detector
Z-axis Jitter Detector
Joint Acceleration Audit
TCP Heatmap Stability Analysis
Visual Sanity Check (3D TCP path)
RoboDK CSV auto-alignment (J1~J7 / TIME)

Theory
------
Residual analysis inspired by NARH framework
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from core.solver_fingerprint import (
  SolverInstabilityFingerprint,
  print_instability_report
)
from core.root_cause import RootCauseClassifier, print_root_cause


# ============================================================
# Robot configs
# ============================================================

ROBOT_CONFIG = {

    "iiwa14": {
        "name": "KUKA LBR iiwa 14 R820",
        "links": [0.36, 0.42, 0.40]
    },

    "iiwa7": {
        "name": "KUKA LBR iiwa 7 R800",
        "links": [0.34, 0.40, 0.36]
    }

}


# ============================================================
# Robust CSV loader
# Supports:
# 1) Clean joint CSV:
#    J1,J2,J3,J4,J5,J6,J7,ERROR,MM_STEP,DEG_STEP,MOVE_ID,TIME
# 2) RoboDK post-processor CSV:
#    Instruction,...,J1 (deg),...,J7 (deg)
# ============================================================


def _read_csv_flexible(path):

    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                rows = [row for row in csv.reader(f) if row and any(cell.strip() for cell in row)]
            if rows:
                return rows
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read CSV: {path}. Last error: {last_error}")


def _normalize_colname(name):

    raw = str(name).strip().replace("\ufeff", "")
    low = raw.lower().strip()

    if re.fullmatch(r"j\s*([1-7])(?:\s*\(.*?\))?", low):
        idx = re.findall(r"[1-7]", low)[0]
        return f"J{idx}"

    if low in {"instruction", "command"}:
        return "INSTRUCTION"

    if low == "time" or low.startswith("time "):
        return "TIME"

    if low == "error":
        return "ERROR"

    if low == "move_id":
        return "MOVE_ID"

    if low == "mm_step":
        return "MM_STEP"

    if low == "deg_step":
        return "DEG_STEP"

    # keep a stable fallback for debug use
    cleaned = re.sub(r"\s+", "_", raw)
    return cleaned


def _rows_to_dataframe(rows):

    header = [_normalize_colname(x) for x in rows[0]]
    width = len(header)

    data = []
    for row in rows[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        data.append(row)

    return pd.DataFrame(data, columns=header)


def _find_joint_columns(columns):

    joint_map = {}
    for c in columns:
        m = re.fullmatch(r"J([1-7])", str(c))
        if m:
            joint_map[int(m.group(1))] = c

    if all(i in joint_map for i in range(1, 8)):
        return [joint_map[i] for i in range(1, 8)]

    return []


def load_robot_csv(path):

    rows = _read_csv_flexible(path)
    df_raw = _rows_to_dataframe(rows)

    # --------------------------------------------------------
    # Case A: RoboDK post-processor CSV with instruction rows
    # --------------------------------------------------------
    source_format = "generic"

    if "INSTRUCTION" in df_raw.columns:
        instr = df_raw["INSTRUCTION"].astype(str).str.strip().str.lower()
        move_mask = instr.isin({
            "move joints",
            "move joint",
            "movej",
            "move l",
            "move linear",
            "movel"
        })
        df_raw = df_raw.loc[move_mask].copy()
        source_format = "robodk_instruction_csv"

    joint_cols = _find_joint_columns(df_raw.columns)

    # --------------------------------------------------------
    # Case B: headerless / numeric-only fallback
    # --------------------------------------------------------
    if not joint_cols:
        try:
            df_num = pd.read_csv(path, comment="#", header=None, encoding="utf-8-sig")
        except Exception:
            df_num = pd.read_csv(path, comment="#", header=None, encoding="latin1")

        if df_num.shape[1] < 7:
            raise ValueError("CSV does not contain enough joint columns for a 7DoF robot")

        df = df_num.iloc[:, :7].copy()
        df.columns = [f"J{i+1}" for i in range(7)]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna().reset_index(drop=True)
        df["frame"] = np.arange(len(df))

        return df, {
            "source_format": "headerless_numeric_csv",
            "time_source": "default_dt",
            "joint_columns": [f"J{i+1}" for i in range(7)]
        }

    # Keep only the useful columns
    keep_cols = list(joint_cols)
    if "TIME" in df_raw.columns:
        keep_cols.append("TIME")
    if "ERROR" in df_raw.columns:
        keep_cols.append("ERROR")

    df = df_raw[keep_cols].copy()

    # Filter error rows if ERROR column exists
    if "ERROR" in df.columns:
        err = pd.to_numeric(df["ERROR"], errors="coerce").fillna(0)
        df = df.loc[err == 0].copy()
        df = df.drop(columns=["ERROR"])

    # Numeric conversion
    for c in joint_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "TIME" in df.columns:
        df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")

    # Drop incomplete rows
    mandatory = list(joint_cols)
    if "TIME" in df.columns:
        # TIME can be missing for some formats; if all TIME missing we fall back later
        if df["TIME"].notna().any():
            mandatory = mandatory + ["TIME"]
    df = df.dropna(subset=mandatory).reset_index(drop=True)

    # Rename joints to canonical names
    rename_map = {joint_cols[i]: f"J{i+1}" for i in range(7)}
    df = df.rename(columns=rename_map)
    df["frame"] = np.arange(len(df))

    if source_format == "generic" and "TIME" in df.columns:
        source_format = "robodk_joint_timed_csv"
    elif source_format == "generic":
        source_format = "joint_csv"

    return df, {
        "source_format": source_format,
        "time_source": "csv_time" if "TIME" in df.columns else "default_dt",
        "joint_columns": [f"J{i+1}" for i in range(7)]
    }


# ============================================================
# Unit utilities
# ============================================================


def deg2rad(df):

    for j in range(7):
        df[f"J{j+1}"] = np.deg2rad(df[f"J{j+1}"])

    return df



def detect_unit(df):

    values = df[[f"J{i+1}" for i in range(7)]].values

    max_val = np.max(np.abs(values))

    if max_val > 6.5:
        return "deg"
    else:
        return "rad"


# ============================================================
# Utility
# ============================================================


def smooth(data):

    if len(data) < 7:
        return data

    return savgol_filter(data, 7, 2)



def resolve_timebase(df, default_dt):

    if "TIME" in df.columns:
        t = pd.to_numeric(df["TIME"], errors="coerce").to_numpy(dtype=float)

        if len(t) >= 2 and np.all(np.isfinite(t)):
            # Strictly monotonic time is ideal; otherwise fall back safely.
            dt_vec = np.diff(t)
            if np.all(dt_vec > 0):
                return t, "TIME"

    return np.arange(len(df), dtype=float) * float(default_dt), "fixed_dt"


# ============================================================
# Forward Kinematics
# ============================================================


def fk_iiwa(joints, links):

    L1, L2, L3 = links

    j1, j2, j3, j4, j5, j6, j7 = joints

    x = (
        L1*np.cos(j1)*np.cos(j2)
        + L2*np.cos(j1)*np.cos(j2+j3)
        + L3*np.cos(j1)*np.cos(j2+j3+j4)
    )

    y = (
        L1*np.sin(j1)*np.cos(j2)
        + L2*np.sin(j1)*np.cos(j2+j3)
        + L3*np.sin(j1)*np.cos(j2+j3+j4)
    )

    z = (
        L1*np.sin(j2)
        + L2*np.sin(j2+j3)
        + L3*np.sin(j2+j3+j4)
    )

    return np.array([x, y, z])



def compute_tcp(df, links):

    tcp = []

    for i in range(len(df)):

        joints = df.iloc[i][[f"J{k+1}" for k in range(7)]].values

        p = fk_iiwa(joints, links)

        tcp.append(p)

    return np.array(tcp)


# ============================================================
# TCP Jump Detector
# ============================================================


def detect_tcp_jump(tcp, threshold=0.01):

    jumps = []

    for i in range(len(tcp)-1):

        d = np.linalg.norm(tcp[i+1]-tcp[i])

        if d > threshold:
            jumps.append((i, d))

    return jumps


# ============================================================
# NARH Residual Engine
# residual = observed_TCP - smoothed_physical_prediction
# ============================================================


def detect_z_jitter(tcp):

    z = tcp[:, 2]

    z_s = smooth(z)

    residual = z - z_s

    jitter = np.std(residual)

    return jitter, residual


# ============================================================
# Joint acceleration audit
# ============================================================


def joint_acceleration(df, time_base):

    acc_list = []

    for j in range(7):

        name = f"J{j+1}"

        theta = smooth(df[name].values)

        vel = np.gradient(theta, time_base)

        a = np.gradient(vel, time_base)

        acc_list.append(a)

    acc_matrix = np.vstack(acc_list).T

    return acc_matrix


# ============================================================
# RoboDK joint-space associator
# NARH-inspired discrete residual for hidden IK branch changes
# ============================================================


def _unwrap_joint_matrix(df):

    q = df[[f"J{i+1}" for i in range(7)]].to_numpy(dtype=float)

    return np.unwrap(q, axis=0)



def _safe_step_dt(time_base):

    if len(time_base) < 2:
        return np.array([], dtype=float)

    dt = np.diff(np.asarray(time_base, dtype=float))

    if not np.all(np.isfinite(dt)):
        dt = np.full_like(dt, np.nanmedian(dt[np.isfinite(dt)]) if np.isfinite(dt).any() else 1.0)

    positive = dt[dt > 1e-9]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    dt = np.where(dt > 1e-9, dt, fallback)

    return dt



def _robodk_compose(u, v, anchor=None, alpha=0.18, beta=0.10):

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    base = u + v

    # Order-sensitive nearest-neighbor coupling:
    # captures sequential branch-switch style interactions between adjacent joints.
    neighbor = np.zeros(7, dtype=float)
    neighbor[:6] = u[:6] * v[1:7] - v[:6] * u[1:7]
    neighbor[6] = np.dot(u[:6], v[:6]) / 6.0

    # Sign-flip penalty highlights hidden IK branch changes even when TCP stays smooth.
    flip = (np.sign(u) != np.sign(v)).astype(float)
    flip_mag = flip * np.minimum(np.abs(u), np.abs(v))

    redundancy_gain = 1.0
    if anchor is not None:
        redundancy_gain += 0.35 * min(1.0, abs(float(anchor[6])) / np.pi)

    return base + redundancy_gain * (alpha * neighbor + beta * np.sign(v - u) * flip_mag)



def robodk_associator_score(df, time_base):

    q = _unwrap_joint_matrix(df)

    if len(q) < 4:
        empty = np.full(len(q), np.nan)
        return {
            "frame_scores": empty,
            "frame_scores_raw": empty.copy(),
            "events": [],
            "threshold": np.nan,
            "peak_frame": None,
            "peak_score": np.nan,
            "peak_raw": np.nan
        }

    dt = _safe_step_dt(time_base)
    dq = np.diff(q, axis=0)
    step = dq / dt[:, None]

    raw_scores = np.full(len(q), np.nan)
    norm_scores = np.full(len(q), np.nan)

    for i in range(len(step) - 2):
        a = step[i]
        b = step[i + 1]
        c = step[i + 2]

        left = _robodk_compose(
            _robodk_compose(a, b, anchor=q[i + 1]),
            c,
            anchor=q[i + 2]
        )
        right = _robodk_compose(
            a,
            _robodk_compose(b, c, anchor=q[i + 2]),
            anchor=q[i + 1]
        )

        associator = left - right
        raw = float(np.linalg.norm(associator))
        motion_scale = float(np.linalg.norm(a) + np.linalg.norm(b) + np.linalg.norm(c) + 1e-9)
        norm = raw / motion_scale

        frame_idx = i + 2
        raw_scores[frame_idx] = raw
        norm_scores[frame_idx] = norm

    valid = norm_scores[np.isfinite(norm_scores)]

    if len(valid) == 0:
        threshold = np.nan
        peak_frame = None
        peak_score = np.nan
        peak_raw = np.nan
        events = []
    else:
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median))) + 1e-9
        threshold = max(0.12, median + 6.0 * mad)

        candidate_idx = np.where(np.isfinite(norm_scores) & (norm_scores > threshold))[0]
        candidate_idx = sorted(candidate_idx, key=lambda idx: norm_scores[idx], reverse=True)

        events = []
        for idx in candidate_idx[:10]:
            events.append({
                "frame": int(idx),
                "score": float(norm_scores[idx]),
                "raw_score": float(raw_scores[idx]),
                "message": "Hidden IK branch-switch risk: joint-space associator spike"
            })

        peak_frame = int(np.nanargmax(norm_scores)) if np.isfinite(norm_scores).any() else None
        peak_score = float(np.nanmax(norm_scores)) if np.isfinite(norm_scores).any() else np.nan
        peak_raw = float(raw_scores[peak_frame]) if peak_frame is not None else np.nan

    return {
        "frame_scores": norm_scores,
        "frame_scores_raw": raw_scores,
        "events": events,
        "threshold": threshold,
        "peak_frame": peak_frame,
        "peak_score": peak_score,
        "peak_raw": peak_raw
    }



def print_associator_report(assoc_result, tcp_mm=None, tcp_hidden_jump_mm=10.0):

    print("\n" + "=" * 50)
    print("RoboDK Joint-Space Associator (NARH)")
    print("=" * 50)

    peak_frame = assoc_result.get("peak_frame")
    peak_score = assoc_result.get("peak_score")
    peak_raw = assoc_result.get("peak_raw")
    threshold = assoc_result.get("threshold")

    if peak_frame is None or not np.isfinite(peak_score):
        print("No valid associator score could be computed.")
        return

    print(f"Peak frame: {peak_frame}")
    print(f"Peak normalized score: {peak_score:.4f}")
    print(f"Peak raw score: {peak_raw:.4f}")
    print(f"Alert threshold: {threshold:.4f}")

    events = assoc_result.get("events", [])
    if not events:
        print("No critical hidden IK branch-switch spike detected.")
        return

    print("\nTop associator events:")
    for ev in events[:5]:
        frame = ev["frame"]
        line = f"Frame {frame}: score={ev['score']:.4f}"

        if tcp_mm is not None and 1 <= frame < len(tcp_mm):
            tcp_step = float(np.linalg.norm(tcp_mm[frame] - tcp_mm[frame - 1]))
            line += f", TCP step={tcp_step:.2f} mm"
            if tcp_step < tcp_hidden_jump_mm:
                line += " -> hidden IK jump risk (TCP still looks smooth)"

        print(line)




# ============================================================
# Heatmap stability analysis
# ============================================================


def tcp_heatmap(tcp, residual, output):

    x = tcp[:, 0]
    y = tcp[:, 1]

    jitter = np.abs(residual) * 1000

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(
        x,
        y,
        c=jitter,
        cmap="inferno",
        s=8
    )

    plt.colorbar(sc, label="Z jitter magnitude (mm)")

    plt.title("TCP Heatmap Stability Analysis")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    plt.savefig(output / "tcp_heatmap.png", dpi=150)

    plt.close()


# ============================================================
# Visualization sanity check
# ============================================================


def plot_tcp_3d(tcp, output):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("TCP Path Sanity Check")

    plt.savefig(output / "tcp_3d_path.png", dpi=150)

    plt.close()


# ============================================================
# Plotters
# ============================================================


def plot_z_jitter(residual, output):

    plt.figure()

    plt.plot(residual)

    plt.title("Z-axis jitter residual")

    plt.savefig(output / "z_jitter.png")

    plt.close()



def plot_joint_acc(acc, output):

    plt.figure()

    for j in range(acc.shape[1]):
        plt.plot(acc[:, j], label=f"J{j+1}")

    plt.grid(True)

    plt.legend()

    plt.title("Joint acceleration (rad/s²)")

    plt.savefig(output / "joint_acc.png")

    plt.close()



def plot_associator(frame_scores, threshold, output):

    plt.figure()
    plt.plot(frame_scores, label="Associator score")

    if np.isfinite(threshold):
        plt.axhline(threshold, linestyle="--", label="Alert threshold")

    plt.grid(True)
    plt.legend()
    plt.title("RoboDK joint-space associator")
    plt.xlabel("Frame")
    plt.ylabel("Normalized score")

    plt.savefig(output / "robodk_associator.png")

    plt.close()


# ============================================================
# Report
# ============================================================


def write_report(robot, frames, jitter, jumps, output, source_format=None, time_mode=None, associator=None):

    text = f"""
SIPA v2.3 Industrial Solver Audit
Robot: {robot}

Frames: {frames}
Input Format: {source_format or 'unknown'}
Time Base: {time_mode or 'unknown'}

TCP Z Jitter
------------
Std amplitude: {jitter*1000:.2f} mm

TCP Jump Events
---------------
{len(jumps)}
"""

    for f, d in jumps[:5]:
        text += f"Frame {f} Jump {d*1000:.2f} mm\n"

    if jitter > 0.001:
        text += "\nDiagnosis: micro oscillation detected\n"

    if len(jumps) > 0:
        text += "Possible solver divergence\n"

    if associator is not None:
        peak_frame = associator.get("peak_frame")
        peak_score = associator.get("peak_score")
        threshold = associator.get("threshold")
        event_count = len(associator.get("events", []))

        text += "\nRoboDK Joint-Space Associator\n"
        text += "---------------------------\n"
        if peak_frame is not None and np.isfinite(peak_score):
            text += f"Peak frame: {peak_frame}\n"
            text += f"Peak normalized score: {peak_score:.4f}\n"
            text += f"Alert threshold: {threshold:.4f}\n"
            text += f"Critical associator events: {event_count}\n"
            if event_count > 0:
                text += "Interpretation: hidden IK branch-switch risk may exist even if TCP looks smooth.\n"
        else:
            text += "No valid associator score computed.\n"

    path = output / "audit_report.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return path


# ============================================================
# Main audit
# ============================================================


def run_audit(input_file, robot, dt, unit, output):

    cfg = ROBOT_CONFIG[robot]

    print("\n" + "="*40)
    print(f"SIPA v2.3 - Auditing: {cfg['name']}")
    print("="*40)
    print("Residual Engine: NARH (Non-Associative Residual Hypothesis)")

    # 1. Load data
    df, meta = load_robot_csv(input_file)
    print(f"[SIPA] Source format: {meta['source_format']}")
    print(f"[SIPA] Joint columns: {', '.join(meta['joint_columns'])}")

    # 2. Unit handling
    if unit == "auto":
        unit = detect_unit(df)
        print(f"[SIPA] Auto detected unit: {unit}")

    if unit == "deg":
        print("[SIPA] Converting deg -> rad")
        df = deg2rad(df)
    else:
        print("[SIPA] Using radians")

    # 3. Time base
    time_base, time_mode = resolve_timebase(df, dt)
    print(f"[SIPA] Time base: {time_mode}")

    # 4. Core Physics Calculations
    tcp = compute_tcp(df, cfg["links"])
    jumps = detect_tcp_jump(tcp)
    jitter, residual = detect_z_jitter(tcp)
    acc = joint_acceleration(df, time_base)
    associator = robodk_associator_score(df, time_base)

    # 5. Solver Instability Fingerprint
    fingerprint = SolverInstabilityFingerprint(
        tcp_threshold_mm=30,
        acc_threshold=300,
        z_threshold_mm=20
    )

    tcp_positions = tcp[:, :3] * 1000
    z_residual_mm = residual * 1000

    instability_events = fingerprint.detect(
        tcp_positions,
        acc,
        z_residual_mm
    )

    # 6. v2.3 Root Cause Classification
    classifier = RootCauseClassifier()
    root_results = classifier.classify(
        tcp_positions,
        acc,
        z_residual_mm
    )

    # 7. Visualization
    plot_tcp_3d(tcp, output)
    plot_z_jitter(residual, output)
    plot_joint_acc(acc, output)
    plot_associator(associator["frame_scores"], associator["threshold"], output)
    tcp_heatmap(tcp, residual, output)

    # 8. Reporting
    report = write_report(
        cfg["name"],
        len(df),
        jitter,
        jumps,
        output,
        source_format=meta["source_format"],
        time_mode=time_mode,
        associator=associator
    )

    # 9. Console Output
    print("\n" + "-"*30)
    print("SIPA v2.3 - INDUSTRIAL DIAGNOSTICS")
    print("="*30)

    print_root_cause(root_results)
    print_associator_report(associator, tcp_mm=tcp_positions)

    if instability_events:
        print_instability_report(instability_events)

    print("Summary Statistics:")
    print(f"TCP jitter (RMS): {round(jitter*1000, 3)} mm")
    print(f"Raw TCP jump frames: {len(jumps)}")
    print(f"Critical Solver Failures: {len(instability_events)}")
    print(f"Associator critical events: {len(associator['events'])}")
    if associator["peak_frame"] is not None and np.isfinite(associator["peak_score"]):
        print(f"Associator peak: frame {associator['peak_frame']} score {associator['peak_score']:.4f}")
    print(f"Detailed Report: {report}")
    print("-"*30)


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--robot", default="iiwa14", choices=["iiwa14", "iiwa7"])
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument(
        "--unit",
        default="auto",
        choices=["auto", "deg", "rad"],
        help="Joint angle unit"
    )

    parser.add_argument("--output", default="outputs")

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    run_audit(
        args.input,
        args.robot,
        args.dt,
        args.unit,
        output
    )


if __name__ == "__main__":
    main()
