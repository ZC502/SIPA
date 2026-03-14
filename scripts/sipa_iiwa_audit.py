#!/usr/bin/env python3

"""
SIPA v2.1 Industrial Specialist
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

Theory
------
Residual analysis inspired by NARH framework
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path


# ============================================================
# Robot configs
# ============================================================

ROBOT_CONFIG = {

    "iiwa14": {
        "name": "KUKA LBR iiwa 14 R820",
        "links": [0.36,0.42,0.40]
    },

    "iiwa7": {
        "name": "KUKA LBR iiwa 7 R800",
        "links": [0.34,0.40,0.36]
    }

}


# ============================================================
# Robust CSV loader
# ============================================================

def load_kuka_csv(path):

    df = pd.read_csv(path,comment="#")

    if df.shape[1] != 7:
        df = pd.read_csv(path,header=None)

    df = df.iloc[:,:7]

    df.columns = [f"J{i+1}" for i in range(7)]

    df["frame"] = np.arange(len(df))

    return df


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

    return savgol_filter(data,7,2)


# ============================================================
# Forward Kinematics
# ============================================================

def fk_iiwa(joints,links):

    L1,L2,L3 = links

    j1,j2,j3,j4,j5,j6,j7 = joints

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

    return np.array([x,y,z])


def compute_tcp(df,links):

    tcp = []

    for i in range(len(df)):

        joints = df.iloc[i][[f"J{k+1}" for k in range(7)]].values

        p = fk_iiwa(joints,links)

        tcp.append(p)

    return np.array(tcp)


# ============================================================
# TCP Jump Detector
# ============================================================

def detect_tcp_jump(tcp):

    jumps = []

    for i in range(len(tcp)-1):

        d = np.linalg.norm(tcp[i+1]-tcp[i])

        if d > 0.002:
            jumps.append((i,d))

    return jumps


# ============================================================
# Z-axis jitter (NARH residual analysis)
# ============================================================

def detect_z_jitter(tcp):

    z = tcp[:,2]

    z_s = smooth(z)

    residual = z - z_s

    jitter = np.std(residual)

    return jitter,residual


# ============================================================
# Joint acceleration audit
# ============================================================

def joint_acceleration(df,dt):

    acc = {}

    for j in range(7):

        name = f"J{j+1}"

        theta = smooth(df[name].values)

        vel = np.gradient(theta,dt)

        a = np.gradient(vel,dt)

        acc[name] = a

    return acc


# ============================================================
# Heatmap stability analysis
# ============================================================

def tcp_heatmap(tcp,residual,output):

    x = tcp[:,0]
    y = tcp[:,1]

    jitter = np.abs(residual)

    plt.figure(figsize=(6,6))

    sc = plt.scatter(
        x,
        y,
        c=jitter,
        cmap="inferno",
        s=8
    )

    plt.colorbar(sc,label="Z jitter magnitude")

    plt.title("TCP Stability Heatmap")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.savefig(output/"tcp_heatmap.png",dpi=150)

    plt.close()


# ============================================================
# Visualization sanity check
# ============================================================

def plot_tcp_3d(tcp,output):

    fig = plt.figure()

    ax = fig.add_subplot(111,projection="3d")

    ax.plot(tcp[:,0],tcp[:,1],tcp[:,2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("TCP Path Sanity Check")

    plt.savefig(output/"tcp_3d_path.png",dpi=150)

    plt.close()


# ============================================================
# Plotters
# ============================================================

def plot_z_jitter(residual,output):

    plt.figure()

    plt.plot(residual)

    plt.title("Z-axis jitter residual")

    plt.savefig(output/"z_jitter.png")

    plt.close()


def plot_joint_acc(acc,output):

    plt.figure()

    for j,a in acc.items():
        plt.plot(a,label=j)

    plt.legend()

    plt.title("Joint acceleration")

    plt.savefig(output/"joint_acc.png")

    plt.close()


# ============================================================
# Report
# ============================================================

def write_report(robot,frames,jitter,jumps,output):

    text = f"""
SIPA v2.1 Industrial Specialist
Robot: {robot}

Frames: {frames}

TCP Z Jitter
------------
Std amplitude: {jitter*1000:.2f} mm

TCP Jump Events
---------------
{len(jumps)}
"""

    for f,d in jumps[:5]:
        text += f"Frame {f} Jump {d*1000:.2f} mm\n"

    if jitter > 0.001:
        text += "\nDiagnosis: micro oscillation detected\n"

    if len(jumps)>0:
        text += "Possible solver divergence\n"

    path = output/"audit_report.txt"

    with open(path,"w") as f:
        f.write(text)

    return path


# ============================================================
# Main audit
# ============================================================

def run_audit(input_file,robot,dt,unit,output):

    cfg = ROBOT_CONFIG[robot]

    print("\nRobot:",cfg["name"])

    df = load_kuka_csv(input_file)

    # unit handling

    if unit == "auto":
        unit = detect_unit(df)
        print("[SIPA] Auto detected unit:",unit)

    if unit == "deg":
        print("[SIPA] Converting deg → rad")
        df = deg2rad(df)
    else:
        print("[SIPA] Using radians")

    tcp = compute_tcp(df,cfg["links"])

    jumps = detect_tcp_jump(tcp)

    jitter,residual = detect_z_jitter(tcp)

    acc = joint_acceleration(df,dt)

    plot_tcp_3d(tcp,output)
    plot_z_jitter(residual,output)
    plot_joint_acc(acc,output)
    tcp_heatmap(tcp,residual,output)

    report = write_report(
        cfg["name"],
        len(df),
        jitter,
        jumps,
        output
    )

    print("\nTCP jitter:",round(jitter*1000,3),"mm")
    print("TCP jumps:",len(jumps))
    print("\nReport:",report)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input",required=True)
    parser.add_argument("--robot",default="iiwa14",choices=["iiwa14","iiwa7"])
    parser.add_argument("--dt",type=float,default=0.01)

    parser.add_argument(
        "--unit",
        default="auto",
        choices=["auto","deg","rad"],
        help="Joint angle unit"
    )

    parser.add_argument("--output",default="outputs")

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(exist_ok=True)

    run_audit(
        args.input,
        args.robot,
        args.dt,
        args.unit,
        output
    )


if __name__ == "__main__":
    main()
