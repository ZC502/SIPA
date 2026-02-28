# SIPA: Spatial Intelligence Physical Audit ⚖️
*A Trajectory-Level Diagnostic Framework for Quantitative Physical Consistency in Spatial AI.*

SIPA is an engine-agnostic auditing protocol designed to provide **Technical Due Diligence (Tech DD)** for generative world models and spatial intelligence systems. By testing the **Non-Associative Residual Hypothesis (NARH)** via octonion formulation, SIPA estimates the onset of physical inconsistency—identifying precisely where visual fluency masks algebraic divergence.

**This means that SIPA is model-agnostic and does not require access to internal states, making it suitable for third-parties to conduct technical due diligence.** 

  **Technical Anchor**: The octonion formulation is used purely as an **algebraic consistency probe** and does not assume any specific force, mass, or friction model. It operates on 7-DoF pose trajectories without requiring simulator internals.SIPA intentionally operates at the kinematic-consistency layer to remain engine-agnostic and broadly deployable.

**Core Methodology: The Non-Associative Residual Hypothesis (NARH)**

SIPA's audit engine is built upon the **Non-Associative Residual Hypothesis (NARH)**. This hypothesis posits that deviations in causal integrity of a 3D trajectory are reflected in the algebraic associativity of its octonion representation. Any "physical hallucination" or causal drift manifests as a non-zero residual in the octonion associator.

- **Theoretical Foundation:**

https://github.com/ZC502/Isaac-Sim-Physical-consistency-plugin?tab=readme-ov-file#non-associative-residual-hypothesis-narh

https://github.com/ZC502/TinyOEKF/blob/master/docs/Continuous_Physics_Solver_for_AI_Wang_Liu.pdf

**Note**:SIPA does not claim that the octonion associator directly models physical forces; rather, it serves as an empirical algebraic probe that is sensitive to causal discontinuities in pose trajectories.

---

🧩 **What SIPA Can Audit**

SIPA is a **trajectory-level physical consistency diagnostic**.
It does **not require source code access or internal simulator states**.

By design, SIPA is compatible with any system that produces spatial motion data.

**Engine-Agnostic Compatibility**

SIPA operates on the final motion output of world models and physics engines, enabling post-hoc physical forensics for:
- **Physics Simulators**: NVIDIA Isaac Sim, MuJoCo, PyBullet, Gazebo
- **Neural World Models**: World Labs Marble, OpenAI Sora, Runway Gen-3 (via pose extraction)
- **Robotic Foundation Models**: any system outputting 7-DoF trajectories
- **Real-World Capture**: OptiTrack, Vicon, or SLAM-based motion sequences

**Supported Data Pathways**

SIPA is designed to audit **trajectory-level physical consistency**.

Different upstream systems provide motion data with varying levels of structural fidelity.

🟢 **Tier 1 — Native Spatial Intelligence (Recommended)**

Systems that expose object-level state directly:
- NVIDIA Isaac Sim
- MuJoCo
- PyBullet / Gazebo
- Robotic telemetry logs
- Motion capture systems

These provide **low-uncertainty pose trajectories**.

🟢 **Tier 2 — Structured World Generators**

Emerging spatial world models such as:
- World Labs **Marble**

While not always emitting CSV natively, these systems generate **programmable 3D world states that** can typically be exported to structured 7-DoF trajectories via post-processing.

✅ object-level identity

✅ temporally coherent motion

✅ high audit fidelity

**SIPA is particularly well-suited for this class of models.**

🟡 **Tier 3 — Pixel Video Models (Experimental)**

Pure video generators such as:
- OpenAI Sora
- Runway Gen-3

require an additional pose-lifting step:
```
text
Video → Pose Estimation → 7-DoF CSV → SIPA
```
⚠️ This pathway introduces vision uncertainty and is considered **experimental / research-grade**.

---

🚀 **Quick Start (Audit Execution)**

SIPA is designed for rapid integration into automated evaluation pipelines (CI/CD) or manual technical audits.

Typical runtime: ~1–3 seconds for a 1k-frame trajectory on a modern laptop CPU (single-threaded).

**1. Environment Setup**

```Bash
git clone https://github.com/your-repo/SIPA.git
cd SIPA
pip install pandas numpy matplotlib
```
**2. Benchmark Execution**

Run the baseline audits to verify system sensitivity:

✅ **Case A: High-Fidelity Consistency**

Audit a smooth, physically solvent trajectory.
```
Bash
python scripts/run_audit.py --input demo/sipa_minimal_trajectory.csv --dt 0.01
# Expected behavior: Rating typically A–B, PIR ≈ 0.85–0.95
```
❌ **Case B: Physical Hallucination (NARH Breach)**
Audit a trajectory with intentional causal collapse (spatial leaps/jitter).
```
Bash
python scripts/run_audit.py --input demo/sipa_corrupted_trajectory.csv --dt 0.01
# Expected behavior: Rating typically D–F, PIR < 0.50, IDO detected
```
**Note:** 
- IDO is designed as an empirical early-warning indicator rather than a formal change-point guarantee.
- As with any post-hoc diagnostic, carefully constructed adversarial trajectories may partially evade detection; SIPA is intended as a high-sensitivity screening tool rather than a formal physical verifier.
- Performance may vary depending on trajectory smoothness, sampling rate, and sensor noise.

**3. Forensic Interpretation**

Outputs are generated in the ```outputs/``` directory:

- ```sipa_audit_pir_evolution.png```: The **Diagnostic Verdict Sheet** featuring the PIR curve, Confidence Envelope, and **Integrity Degradation Onset (IDO)**.
- **Terminal Summary**: A quantitative credit-style rating for rapid executive review.

---

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
4. Violations are reported but do not automatically abort the audit.

**Disclaimer**: SIPA provides **trajectory-level physical consistency diagnostics**. It detects causal drift and structural instabilities without requiring internal access to the engine's force/torque solvers.

**Scope Note**: 

1.SIPA evaluates kinematic and algebraic consistency of observed trajectories. It does not attempt to fully reconstruct underlying physical forces or guarantee real-world safety.

2.All trajectories are expected to be expressed in consistent SI units; inconsistent scaling may affect the physical debt term.

**Developer Integration (Quick Export)**

If you are using **Isaac Sim (Omniverse)** or **custom Python trackers**, use this helper to export trajectories compatible with the SIPA Auditor:

```python
def export_sipa_csv(path, positions, quaternions):
    """
    Export trajectory to SIPA-compatible CSV format.
    :param positions: (T, 3) numpy array
    :param quaternions: (T, 4) numpy array [qx, qy, qz, qw]
    Expected dtype: float32 or float64
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

Optional flags:

--verbose      Enable detailed logs  
--branding     Enable formatted header output

```

---

**Physical Integrity Rating (PIR)**

SIPA introduces the **Physical Integrity Rating (PIR)**, a heuristic composite indicator designed to quantify the causal reliability of motion trajectories. PIR evaluates whether a world model is "physically solvent" or accumulating "kinetic debt."

**The Metric**
$$PIR = Q_{\text{data}} \times (1 - D_{\text{phys}})$$
- $Q_{\text{data}}$ **(Data Quality)**: Measures input integrity (SNR, normalization, temporal jitter).
- $D_{\text{phys}}$ **(Physical Debt)**:  Log-normalized residual derived from the Octonion Associator, testing the NARH limits.
- $PIR \in [0, 1]$: Higher indicates higher physical fidelity.

📊 Credit Rating Scale
| PIR Score | Rating | Label          | Operational Meaning                                                                 |
|-----------|--------|----------------|-------------------------------------------------------------------------------------|
| ≥ 0.85    | A      | High Integrity | Reliable for industrial simulation and safety-critical AI.                         |
| ≥ 0.70    | B      | Acceptable     | Generally consistent; minor numerical drift detected.                               |
| ≥ 0.50    | C      | Speculative    | "Visual plausibility maintained, but causal logic is shaky."                       |
| ≥ 0.30    | D      | High Risk      | "Elevated physical debt; prone to ""hallucinations"" under stress."                |
| < 0.30    | F      | Critical       | Physical bankruptcy; trajectory violates fundamental causality.                    |

**Note**: 

1.PIR is a **diagnostic risk indicator**. It serves as a "stress test" for neural world models (e.g., World Labs, Sora, FSD) rather than a formal proof of physical validity.

2.Unlike conventional smoothness or jerk-based metrics, PIR is sensitive to non-associative temporal inconsistencies that may remain visually smooth but algebraically unstable.

3.PIR incorporates a data-quality term (Q_data) to reduce sensitivity to high-frequency measurement noise; however, extremely noisy trajectories may require standard smoothing pre-processing.

---

🔭 **Future Directions**
SIPA currently focuses on **trajectory-level physical consistency**.

Future extensions may include:

🚗 **Autonomous Driving Forensics**

Applying SIPA-style causal auditing to:
- end-to-end driving policies
- planning stack trajectories
- long-horizon closed-loop rollouts

Potential targets include systems similar to:
- systems such as Tesla FSD
- embodied driving world models
- neural simulation stacks

The long-term goal is to provide **post-hoc physical sanity checks** for safety-critical spatial intelligence systems.

---

💼 **Business Impact & Value Proposition**

SIPA transforms qualitative "visual plausibility" into quantitative **Physical Solvency**. For stakeholders in Spatial AI and Robotics, SIPA provides four critical layers of value:

**1. Technical Due Diligence (Tech DD)**

For venture capital and corporate development teams, SIPA serves as a **"quantitative consistency probe for world models**. It quantifies whether a startup's generative video (e.g., World Labs) possesses true causal depth or is merely "hallucinating" pixel-perfect but physically impossible motion.

**2. Risk Mitigation in Sim-to-Real (Sim2Real)**

Training robots in flawed simulations costs millions in hardware damage. SIPA identifies **Kinetic Debt** in simulation environments (NVIDIA Isaac, MuJoCo) before policy deployment, drastically reducing the failure rate in real-world transitions.

**3. Automated Quality Assurance (QA) for Smart Spaces**

In industrial digital twins and smart warehouses, SIPA acts as an **Automated Auditor**. It continuously monitors 7-DoF telemetry to flag structural anomalies or sensor drifts that violate the **Non-Associative Residual Hypothesis (NARH)**.

**4. Regulatory & Safety Compliance**

As Spatial Intelligence moves towards safety-critical sectors (Autonomous Driving, Surgical Robotics), SIPA provides a standardized, **non-intrusive audit trail** to certify that an AI model's outputs adhere to fundamental algebraic constraints of the physical world.

---

📢 **For the Community: Audit the World**

Is that Spatial AI demo physically real or just a "visual hallucination"? Don't take their word for it. **Run the Audit**.

SIPA empowers every researcher and enthusiast to quantify the "Physical Debt" of any world model. Whether it's a high-budget simulation or a generative video, just export the trajectory and get the **PIR (Physical Integrity Rating)**.

Let's build a transparent baseline for the future of Spatial Intelligence.

---

⚖️ **Intellectual Property & Commercial Licensing**

This repository introduces a novel paradigm in physical auditing based on Non-Associative Octonion Manifolds.
- Intellectual Property: All core algorithms, including the NARH (Non-Associative Residual Hypothesis) and its implementation via octonion associators, are the proprietary intellectual property of the author (Patent Filing in Preparation).
- Academic Use: Non-commercial research and academic evaluation are permitted. Please cite this repository and the referenced works accordingly.
- Commercial/Production Use: Implementation of SIPA protocols in any revenue-generating simulation pipeline, robotic foundation model training, or commercial world model deployment (including data augmentation for LLM/Video-AI) requires a Commercial License.
- Strategic Inquiries: For inquiries regarding full IP acquisition, strategic investment, or dedicated implementation support for enterprise-grade spatial intelligence, please contact:
Strategic Inquiry & Business Contact: > 📧 [liuzc19761204@gmail.com]

---
📚 **Recommended Citation**

If SIPA is integrated into your research or technical due diligence pipeline, please cite:

**SIPA: Spatial Intelligence Physical Audit (2026)**.
