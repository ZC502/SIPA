# SIPA: Spatial Intelligence Physical Audit ⚖️
*A Diagnostic Framework for Quantitative Physical Consistency in Spatial AI.*

SIPA is an engine-agnostic auditing protocol designed to provide **Technical Due Diligence (Tech DD)** for generative world models and spatial intelligence systems. As visual fidelity in models reaches a "plausibility plateau," SIPA introduces a rigorous metric to quantify the underlying **Causal Integrity**.

By analyzing **Non-Associative Residual Dynamics** via octonion formulation, SIPA estimates the onset of physical inconsistency—identifying precisely where visual fluency masks algebraic divergence.

  **Core Principle**: The octonion formulation is used purely as an **algebraic consistency probe** and does not assume any specific force, mass, or friction model. It operates purely on pose trajectories without requiring access to simulator internals.

🔍 **What SIPA Can Audit**

SIPA is a **trajectory-level** diagnostic tool. It does not require source code access or internal simulator states. By design, SIPA is compatible with any system that generates spatial motion data.

🧩 **Engine-Agnostic Compatibility**

SIPA operates on the final output of world models and physics engines. It acts as a **Universal Physical Judge** for:

- **Physics Simulators**: NVIDIA Isaac Sim, MuJoCo, PyBullet, Gazebo.
- **Neural World Models**: World Labs (Marble), OpenAI Sora, Runway Gen-3 (via pose extraction).
- **Robotic Foundation Models**: Any model outputting 7-DoF end-effector or base trajectories.
- **Real-world Capture**: OptiTrack, Vicon, or SLAM-based motion sequences.

🎥 **Video-Based Auditing Pipeline**

For video-only sources (where a CSV trajectory is not natively provided), SIPA enables **post-hoc forensics** via a two-step pipeline:

1. **Pose Extraction**: Use external tools (e.g., COLMAP, Foundation Pose, or VitPose) to extract the 7-DoF trajectory.

2. **SIPA Audit**: Ingest the extracted CSV into run_audit.py to quantify the physical validity of the video's motion.

📄 **Supported Input Matrix**

| Source Category | Status      | Requirement              | Use Case                          |
|-----------------|-------------|--------------------------|-----------------------------------|
| **Simulators**      | ✅ **Native**   | Export 7-DoF Pose CSV    | RL Training, Simulation Sanity    |
| **Motion Capture**  | ✅ **Native**   | Export Pose Sequence     | Real-vs-Sim Benchmarking          |
| **Video AI**        | ⚠️ **Indirect** | Requires Pose Extraction | **Tech DD for World Models**          |
| **Robotic Logs**    | ✅ **Native**   | Telemetry/Odom CSV       | Deployment Safety Audit           |

---

🚀 **Quick Start (Audit Execution)**

SIPA is designed for rapid integration into automated evaluation pipelines (CI/CD) or manual technical audits.

**1. Environment Setup**

```Bash
git clone https://github.com/your-repo/SIPA.git
cd SIPA
pip install pandas numpy matplotlib
```
**2. Benchmark Execution**

Run the baseline audits to verify system sensitivity:

✅ **Baseline A: High-Fidelity Consistency**

Audit a trajectory with stable physical debt.
```
Bash
python scripts/run_audit.py --input demo/sipa_minimal_trajectory.csv --dt 0.01
# Expected Output: FINAL RATING [A/B], PIR ≈ 0.85–0.95, No IDO marker.
```
❌ **Baseline B: Causal Integrity Breach**
Audit a trajectory with intentional physical hallucinations (spatial leaps/pose jitter).
```
Bash
python scripts/run_audit.py --input demo/sipa_corrupted_trajectory.csv --dt 0.01
# Expected Output: FINAL RATING [D/F], PIR < 0.50, RED Onset Marker (IDO) detected.
```
**3. Forensic Interpretation**

Outputs are generated in the ```outputs/``` directory:

- ```sipa_audit_pir_evolution.png```: The **Diagnostic Verdict Sheet** featuring the PIR curve, Confidence Envelope, and **Integrity Degradation Onset (IDO)**.
- **Terminal Summary**: A quantitative credit-style rating for rapid executive review.

---

🔬 **Physical Integrity Rating (PIR)**

SIPA introduces the **PIR**, a **diagnostic composite indicator** designed to quantify the causal reliability of motion trajectories. PIR evaluates whether a world model is "physically solvent" or accumulating "kinetic debt."

**The Metric**

$$PIR = Q_{\text{data}} \times (1 - D_{\text{phys}})$$

- $Q_{\text{data}}$ (**Data Quality**): Algebraic SNR, normalization integrity, and temporal monotonicity.
- $D_{\text{phys}}$ (**Physical Debt**): Log-normalized non-associative residual derived from the Octonion Associator.
- $PIR \in [0, 1]$: Higher indicates superior structural consistency.

**SIPA Audit Protocol (v1.0 Schema)**

To ensure engine-agnostic physical auditing, SIPA follows a strict **7-DoF Pose Trajectory Schema**. Any motion trajectory extracted from **NVIDIA Isaac Sim, World Labs Marble**, or **Real-world Video** can be audited if it conforms to this standard.

📄 **CSV Input Format**

The auditor expects a CSV file with the following 7 required columns:
| **Column**      | **Unit** | **Description**                          |
|-----------------|----------|------------------------------------------|
| x, y, z         | meters   | 3D Position in World Frame (SI Units)    |
| qx, qy, qz, qw  | unitless | Normalized Quaternion (Orientation)      |

**Minimal Example** (```trajectory.csv```):
```
x,y,z,qx,qy,qz,qw
0.000,0.000,0.500,0.0,0.0,0.0,1.0
0.002,0.000,0.498,0.001,0.0,0.0,0.999
0.006,0.001,0.493,0.002,0.001,0.0,0.999
```
**Temporal Requirements**
- **Standard Mode**: Use ```--dt``` (e.g., ```0.02```) for uniform sampling.
- **Precision Mode**: Add a ```t ```(seconds) column for non-uniform timestamps.

**Data Integrity Rules (The "SIPA Guardrail")**
SIPA automatically validates your data before auditing: 
1. **Normalization**: Quaternions must satisfy $||q|| \in [0.999, 1.001]$.
2. **Continuity**: Significant spatial leaps without corresponding velocity will be flagged as **Data Artifacts**.
3. **Monotonicity**: Timestamps must be strictly increasing.

**Disclaimer**: SIPA provides **trajectory-level physical consistency diagnostics**. It detects causal drift and structural instabilities without requiring internal access to the engine's force/torque solvers.

**Developer Integration (Quick Export)**

If you are using **Isaac Sim (Omniverse)** or **custom Python trackers**, use this helper to export trajectories compatible with the SIPA Auditor:

```python
def export_sipa_csv(path, positions, quaternions):
    """
    Export trajectory to SIPA-compatible CSV format.
    :param positions: (T, 3) numpy array
    :param quaternions: (T, 4) numpy array [qx, qy, qz, qw]
    """
    import pandas as pd
    import numpy as np

    assert positions.shape[1] == 3, "Positions must be (T, 3)"
    assert quaternions.shape[1] == 4, "Quaternions must be (T, 4)"

    df = pd.DataFrame({
        "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
        "qx": quaternions[:, 0], "qy": quaternions[:, 1], 
        "qz": quaternions[:, 2], "qw": quaternions[:, 3],
    })
    df.to_csv(path, index=False)
    print(f"[SIPA] Trajectory exported to {path}")
```

---

**Physical Integrity Rating (PIR)**

SIPA introduces the **Physical Integrity Rating (PIR)**, a heuristic composite indicator designed to quantify the causal reliability of motion trajectories. PIR evaluates whether a world model is "physically solvent" or accumulating "kinetic debt."

**The Metric**
$$PIR = Q_{\text{data}} \times (1 - D_{\text{phys}})$$
- $Q_{\text{data}}$ **(Data Quality)**: Measures input integrity (SNR, normalization, temporal jitter).
- $D_{\text{phys}}$ **(Physical Debt)**: Log-normalized non-associative residual ($Octonion \ Associator$).
- $PIR \in [0, 1]$: Higher indicates higher physical fidelity.

📊 Credit Rating Scale
| PIR Score | Rating | Label          | Operational Meaning                                                                 |
|-----------|--------|----------------|-------------------------------------------------------------------------------------|
| ≥ 0.85    | A      | High Integrity | Reliable for industrial simulation and safety-critical AI.                         |
| ≥ 0.70    | B      | Acceptable     | Generally consistent; minor numerical drift detected.                               |
| ≥ 0.50    | C      | Speculative    | "Visual plausibility maintained, but causal logic is shaky."                       |
| ≥ 0.30    | D      | High Risk      | "Elevated physical debt; prone to ""hallucinations"" under stress."                |
| < 0.30    | F      | Critical       | Physical bankruptcy; trajectory violates fundamental causality.                    |

**Note**: PIR is a **diagnostic risk indicator**. It serves as a "stress test" for neural world models (e.g., World Labs, Sora, FSD) rather than a formal proof of physical validity.

---
📚 Recommended Citation
If SIPA is integrated into your research or technical due diligence pipeline, please cite:

**SIPA: Spatial Intelligence Physical Audit (2026)**.
