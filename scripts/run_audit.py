#!/usr/bin/env python3
"""
SIPA Auditor v1.3
Spatial Intelligence Physical Audit

New in v1.3
-----------
✔ Residual spike visualization
✔ PIR evolution plot
✔ 3D trajectory visualization
✔ anomaly frame highlighting
✔ research-grade diagnostic outputs
"""

import argparse
import json
import sys
import random
import platform
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# Deterministic seed
# ------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------
# Industrial CSV Loader
# ------------------------------------------------

TIME_ALIASES = ["frame","timestamp","time","t"]

SPATIAL_COLUMNS = ["x","y","z","qx","qy","qz","qw"]


def load_trajectory(csv_path: Path):

    df = pd.read_csv(
        csv_path,
        comment="#",
        skip_blank_lines=True,
        engine="c",
        low_memory=False
    )

    if len(df)==0:
        raise ValueError("CSV contains zero rows")

    # detect time column
    time_col=None
    for c in TIME_ALIASES:
        if c in df.columns:
            time_col=c
            break

    if time_col is None:
        raise ValueError("No time column detected")

    if time_col!="frame":
        df=df.rename(columns={time_col:"frame"})

    # check spatial columns
    for c in SPATIAL_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    # numeric enforcement
    numeric_cols=["frame"]+SPATIAL_COLUMNS
    for c in numeric_cols:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    if df[numeric_cols].isnull().any().any():
        raise ValueError("Non numeric values detected")

    df=df.sort_values("frame").reset_index(drop=True)

    return df


# ------------------------------------------------
# Physics validation
# ------------------------------------------------

def validate_physics(df):

    score=1.0
    issues=[]

    qnorm=np.sqrt(
        df.qx**2+df.qy**2+df.qz**2+df.qw**2
    )

    if not np.allclose(qnorm,1,atol=1e-2):
        score-=0.2
        issues.append("Quaternion normalization drift")

    if not np.all(np.diff(df.frame.values)>0):
        score-=0.2
        issues.append("Frame discontinuity")

    score=max(score,0)

    return score,issues


# ------------------------------------------------
# Residual computation
# ------------------------------------------------

def compute_residual(df,dt):

    pos=df[["x","y","z"]].values

    vel=np.diff(pos,axis=0)/dt
    acc=np.diff(vel,axis=0)/dt

    residual=np.linalg.norm(acc,axis=1)

    return residual


# ------------------------------------------------
# PIR
# ------------------------------------------------

def compute_pir(residual):

    pir=np.exp(-np.mean(residual)/10)

    return float(pir)


# ------------------------------------------------
# Anomaly detection
# ------------------------------------------------

def detect_anomaly(residual,df):

    idx=int(np.argmax(residual))

    bad_frame=int(df.iloc[idx+2]["frame"])

    max_debt=float(residual[idx])

    return bad_frame,max_debt,idx


# ------------------------------------------------
# Visualization
# ------------------------------------------------

def plot_residual(residual,idx,output_dir):

    plt.figure(figsize=(8,4))

    plt.plot(residual,label="Residual Acceleration")

    plt.axvline(idx,color="red",linestyle="--",label="Anomaly")

    plt.xlabel("Frame")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Residual Physics Spike")

    plt.legend()
    plt.tight_layout()

    path=output_dir/"residual_spike.png"

    plt.savefig(path,dpi=150)
    plt.close()

    return path


def plot_pir_evolution(residual,output_dir):

    pir_series=np.exp(-residual/10)

    plt.figure(figsize=(8,4))

    plt.plot(pir_series)

    plt.xlabel("Frame")
    plt.ylabel("Local PIR")

    plt.title("PIR Evolution")

    plt.tight_layout()

    path=output_dir/"pir_evolution.png"

    plt.savefig(path,dpi=150)
    plt.close()

    return path


def plot_trajectory(df,bad_frame,output_dir):

    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure(figsize=(6,6))

    ax=fig.add_subplot(111,projection="3d")

    ax.plot(df.x,df.y,df.z,label="Trajectory")

    bad=df[df.frame==bad_frame]

    if len(bad)>0:
        ax.scatter(
            bad.x,
            bad.y,
            bad.z,
            color="red",
            s=80,
            label="Anomaly"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("3D Trajectory")

    ax.legend()

    path=output_dir/"trajectory_3d.png"

    plt.tight_layout()
    plt.savefig(path,dpi=150)
    plt.close()

    return path


# ------------------------------------------------
# Rating
# ------------------------------------------------

def rating(pir):

    if pir>0.9: return "A"
    if pir>0.75: return "B"
    if pir>0.6: return "C"
    return "F"


# ------------------------------------------------
# Diagnostic
# ------------------------------------------------

def diagnostic(pir,max_debt,bad_frame):

    if pir>0.9:
        return "PASS: Physically plausible trajectory"

    if max_debt>5000:
        return f"CRITICAL: teleportation-like jump at frame {bad_frame}"

    if max_debt>500:
        return f"WARNING: extreme acceleration spike at frame {bad_frame}"

    return "MODERATE: minor physics inconsistency"


# ------------------------------------------------
# Report
# ------------------------------------------------

def write_report(
    output_dir,input_file,df,pir,rating_label,
    validator_score,issues,bad_frame,max_debt
):

    report=f"""
SIPA Physical Audit Report
==========================

Input trajectory : {input_file}

Frames           : {len(df)}
Validator score  : {validator_score:.3f}

PIR Score        : {pir:.3f}
Rating           : {rating_label}

Max Physics Debt : {max_debt:.2f} m/s^2
Anomaly Frame    : {bad_frame}

Diagnostic
----------
{diagnostic(pir,max_debt,bad_frame)}

Validator Issues
----------------
"""

    if issues:
        for i in issues:
            report+=f"- {i}\n"
    else:
        report+="None\n"

    path=output_dir/"audit_report.txt"

    with open(path,"w") as f:
        f.write(report)

    return path


# ------------------------------------------------
# Metadata
# ------------------------------------------------

def write_metadata(output_dir,input_file,dt,pir):

    meta={
        "timestamp":datetime.datetime.utcnow().isoformat(),
        "python":platform.python_version(),
        "platform":platform.platform(),
        "input_file":str(input_file),
        "dt":dt,
        "pir":pir
    }

    path=output_dir/"audit_metadata.json"

    with open(path,"w") as f:
        json.dump(meta,f,indent=2)

    return path


# ------------------------------------------------
# CLI
# ------------------------------------------------

def parse_args():

    parser=argparse.ArgumentParser(
        description="SIPA Physical Auditor v1.3"
    )

    parser.add_argument("--input",required=True)
    parser.add_argument("--dt",type=float,default=1/60)
    parser.add_argument("--output",default="outputs")
    parser.add_argument("--seed",type=int,default=42)

    return parser.parse_args()


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    args=parse_args()

    set_seed(args.seed)

    input_csv=Path(args.input)
    output_dir=Path(args.output)

    output_dir.mkdir(parents=True,exist_ok=True)

    print("\n[SIPA] Loading trajectory")

    df=load_trajectory(input_csv)

    validator_score,issues=validate_physics(df)

    print("[SIPA] Computing residuals")

    residual=compute_residual(df,args.dt)

    pir=compute_pir(residual)

    rating_label=rating(pir)

    bad_frame,max_debt,idx=detect_anomaly(residual,df)

    # visualizations

    res_plot=plot_residual(residual,idx,output_dir)

    pir_plot=plot_pir_evolution(residual,output_dir)

    traj_plot=plot_trajectory(df,bad_frame,output_dir)

    report_path=write_report(
        output_dir,input_csv,df,pir,rating_label,
        validator_score,issues,bad_frame,max_debt
    )

    meta_path=write_metadata(
        output_dir,input_csv,args.dt,pir
    )

    print("\n[SIPA RESULT]")
    print("PIR:",round(pir,3))
    print("Rating:",rating_label)
    print("Bad Frame:",bad_frame)
    print("Max Physics Debt:",round(max_debt,2))

    print("\nGenerated files:")
    print(report_path)
    print(meta_path)
    print(res_plot)
    print(pir_plot)
    print(traj_plot)
    print()


if __name__=="__main__":
    main()
