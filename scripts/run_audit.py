#!/usr/bin/env python3

"""
SIPA v2.0 Robotics Auditor

Modes
-----
cartesian : original SIPA trajectory audit
kuka      : industrial robot joint-space audit

New Industrial Features
-----------------------
Joint acceleration physics debt
Joint limits validation
Savitzky-Golay denoising
Axis synchronization audit
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path


# ------------------------------------------------
# KUKA JOINT LIMITS (approx industrial defaults)
# ------------------------------------------------

JOINT_LIMITS = {
    "J1": (-185,185),
    "J2": (-120,120),
    "J3": (-170,170),
    "J4": (-185,185),
    "J5": (-120,120),
    "J6": (-350,350),
    "J7": (-350,350),
}


# ------------------------------------------------
# Loader for KUKA CSV
# ------------------------------------------------

def load_kuka_csv(path):

    df = pd.read_csv(path, header=None)

    if df.shape[1] != 7:
        raise ValueError("KUKA CSV must have 7 columns")

    df.columns = [f"J{i+1}" for i in range(7)]

    df["frame"] = np.arange(len(df))

    return df


# ------------------------------------------------
# Joint limit validation
# ------------------------------------------------

def validate_joint_limits(df):

    violations = []

    for j,(low,high) in JOINT_LIMITS.items():

        bad = df[(df[j] < low) | (df[j] > high)]

        if len(bad) > 0:
            violations.append(j)

    return violations


# ------------------------------------------------
# Smoothing
# ------------------------------------------------

def smooth_signal(data):

    if len(data) < 7:
        return data

    return savgol_filter(data,7,2)


# ------------------------------------------------
# Compute joint acceleration
# ------------------------------------------------

def compute_joint_acc(df,dt):

    acc = {}

    for j in [f"J{i+1}" for i in range(7)]:

        theta = smooth_signal(df[j].values)

        vel = np.gradient(theta,dt)

        a = np.gradient(vel,dt)

        acc[j] = a

    return acc


# ------------------------------------------------
# Physics debt
# ------------------------------------------------

def compute_debt(acc):

    max_debt = 0
    bad_joint = None

    for j,a in acc.items():

        m = np.max(np.abs(a))

        if m > max_debt:
            max_debt = m
            bad_joint = j

    return max_debt,bad_joint


# ------------------------------------------------
# Axis synchronization
# ------------------------------------------------

def axis_sync_score(acc):

    peaks = []

    for a in acc.values():

        peaks.append(np.argmax(np.abs(a)))

    peaks = np.array(peaks)

    return np.std(peaks)


# ------------------------------------------------
# Visualization
# ------------------------------------------------

def plot_joint_acc(acc,output):

    plt.figure(figsize=(10,5))

    for j,a in acc.items():
        plt.plot(a,label=j)

    plt.title("Joint Acceleration")
    plt.xlabel("Frame")
    plt.ylabel("deg/s^2")

    plt.legend()

    p = output/"joint_acceleration.png"

    plt.savefig(p,dpi=150)

    plt.close()

    return p


# ------------------------------------------------
# Report
# ------------------------------------------------

def write_report(
    output,
    frames,
    debt,
    joint,
    limit_violations,
    sync_score
):

    if debt > 5000:
        severity = "CRITICAL"
    elif debt > 2000:
        severity = "WARNING"
    else:
        severity = "OK"

    text = f"""
SIPA Robotics Audit Report
==========================

Frames: {frames}

Max Joint Acceleration Debt: {debt:.2f} deg/s^2
Violation Joint: {joint}

Axis Sync Score: {sync_score:.2f}

Joint Limit Violations: {limit_violations}

Diagnosis
---------
{severity}: Joint acceleration anomaly detected
"""

    p = output/"audit_report.txt"

    with open(p,"w") as f:
        f.write(text)

    return p


# ------------------------------------------------
# KUKA AUDIT
# ------------------------------------------------

def run_kuka(path,dt,output):

    print("\n[SIPA] KUKA mode")

    df = load_kuka_csv(path)

    limit_violations = validate_joint_limits(df)

    acc = compute_joint_acc(df,dt)

    debt,joint = compute_debt(acc)

    sync = axis_sync_score(acc)

    plot_joint_acc(acc,output)

    report = write_report(
        output,
        len(df),
        debt,
        joint,
        limit_violations,
        sync
    )

    print("\nRESULT")
    print("Frames:",len(df))
    print("Max Debt:",round(debt,2),"deg/s^2")
    print("Joint:",joint)
    print("Axis Sync Score:",round(sync,2))
    print("Joint Limits Violated:",limit_violations)

    print("\nReport:",report)


# ------------------------------------------------
# CLI
# ------------------------------------------------

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--input",required=True)
    p.add_argument("--mode",default="kuka")
    p.add_argument("--dt",type=float,default=0.01)
    p.add_argument("--output",default="outputs")

    return p.parse_args()


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    args = parse_args()

    path = Path(args.input)

    output = Path(args.output)

    output.mkdir(exist_ok=True)

    if args.mode == "kuka":
        run_kuka(path,args.dt,output)

    else:
        print("Cartesian mode not included in this snippet")


if __name__ == "__main__":
    main()
