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
