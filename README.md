# **SIPA（Spatial Intelligence Physical Audit）**

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
