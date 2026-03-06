# SIPA: Spatial Intelligence Physical Audit ⚖️
*A Trajectory-Level Diagnostic Framework for Quantitative Physical Consistency in Spatial AI.*

SIPA has **zero heavy dependencies** and runs entirely on CPU.

⏱ Typical runtime: **< 3 seconds for a 1000-frame trajectory**

---

# 🚀 Quick Start (30-Second Demo)

Clone the repository and run the baseline audit examples.

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ZC502/SIPA.git
cd SIPA
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
Recommended Python version
```
Python >= 3.8
```

---

# 🧪 One-Command Demo

Run the full SIPA demo:
```Bash
bash scripts/run_demo.sh
```
This runs two trajectories:

| Case        | Description                    | Expected Result |
| ----------- | ------------------------------ | --------------- |
| ✅ Normal    | Physically consistent motion   | PIR stable      |
| ❌ Corrupted | Injected spatial hallucination | PIR collapse    |

Expected terminal output
```
[1/2] Running normal trajectory audit...
[SIPA] Final PIR: 0.91
[SIPA] Rating: A

[2/2] Running corrupted trajectory audit...
[SIPA] Final PIR: 0.32
[SIPA] Rating: D
```

Output figure:

```
outputs/sipa_audit_pir_evolution.png
```

---

# 📊 Manual Audit (Step-By-Step)

You can also run the auditor manually.

---

### ✅ Case A — Physically Consistent Trajectory
```
python scripts/run_audit.py \
    --input demo/sipa_minimal_trajectory.csv \
    --dt 0.01 \
    --branding
```
Expected result
```
FINAL RATING: A/B
PIR ≈ 0.85 – 0.95
No IDO marker detected
```

---

# 📦 Output Artifacts

All results are saved to
```
outputs/
```
Generated files include
```
sipa_audit_pir_evolution.png
```

This diagnostic sheet visualizes
- PIR evolution
- confidence envelope
- Integrity Degradation Onset (IDO)

---

# 📄 Input Format (7-DoF Pose CSV)

SIPA operates on pose trajectories with the following columns
```
x,y,z,qx,qy,qz,qw
```
Example
```
x,y,z,qx,qy,qz,qw
0.00,0.00,0.50,0,0,0,1
0.01,0.00,0.50,0,0,0,1
0.02,0.00,0.50,0,0,0,1
```
Where
- (x,y,z) = position in meters
- (qx,qy,qz,qw) = unit quaternion rotation

---

# 🧠 Architecture Overview

SIPA evaluates the **physical consistency of spatial trajectories** using a lightweight algebraic diagnostic pipeline.

```markdown
Trajectory (7-DoF CSV)
          │
          ▼
Residual Audit
(Octonion Associator)
          │
          ▼
Physical Debt Accumulation
(Log-normal residual integration)
          │
          ▼
Physical Integrity Rating (PIR)
          │
          ▼
Integrity Degradation Onset (IDO)
```
```
Input  : Pose trajectory (x,y,z,qx,qy,qz,qw)
Output : PIR score + diagnostic visualization
Runtime: < 3 seconds for ~1k frames on CPU
```

**Pipeline Stages**

| Stage                 | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| **Residual Audit**    | Computes octonion associator residuals to detect algebraic inconsistency |
| **Debt Accumulation** | Integrates residuals into a cumulative physical debt signal              |
| **PIR Estimation**    | Converts debt and data quality into a normalized integrity score         |
| **IDO Detection**     | Identifies the onset of structural physical collapse                     |

The pipeline is **engine-agnostic** and requires only motion trajectories.

No simulator internals are required.

---

# 🧪 Physics Hallucination Benchmark

SIPA includes minimal benchmark trajectories to demonstrate detection of **physical hallucinations**.

| Dataset | Description | Frames | Expected PIR | Expected Rating |
|-------|-------------|-------|-------------|----------------|
| `sipa_minimal_trajectory.csv` | Smooth physically consistent motion | ~1000 | 0.85 – 0.95 | A / B |
| `sipa_corrupted_trajectory.csv` | Injected spatial jitter / teleportation | ~1000 | < 0.50 | D / F |

Run benchmark:

```bash
bash scripts/run_demo.sh
```
Expected behavior:
| Metric        | Normal Motion | Corrupted Motion   |
| ------------- | ------------- | ------------------ |
| PIR Stability | Stable        | Collapse           |
| Residual Debt | Low           | Rapid accumulation |
| IDO Marker    | None          | Triggered          |

This benchmark illustrates the **Non-Associative Residual Hypothesis (NARH):**

physically inconsistent trajectories accumulate residual algebraic error that cannot be reconciled under non-associative composition.

---

# 🧪 Adversarial Trajectory Test

A common failure mode of generative world models is **physically inconsistent motion that remains visually smooth**.

These trajectories may appear plausible to humans but violate deeper **causal structure**.

SIPA includes an adversarial demonstration to illustrate this phenomenon.

---

## Concept

In the adversarial trajectory:

- Motion appears **smooth and continuous**
- No obvious teleportation occurs
- Position curves remain visually plausible

However:

- Hidden temporal inconsistency is injected
- Small orientation drift accumulates
- Algebraic associativity is violated

This produces **residual debt accumulation** detectable by SIPA.

---

## Visual vs Physical Consistency

| Property | Human Visual Inspection | SIPA Physical Audit |
|--------|--------------------------|--------------------|
| Motion Smoothness | ✓ Appears smooth | ✓ Smooth |
| Teleportation | ✗ None visible | ✓ None |
| Orientation Drift | Hard to detect | Detected |
| Algebraic Consistency | Not observable | Violated |
| PIR Stability | Appears normal | **Collapses over time** |

---

## Running the Test

If an adversarial trajectory is available:

```bash
python scripts/run_audit.py \
  --input demo/sipa_adversarial_trajectory.csv \
  --dt 0.01 \
  --branding
```
Expected behavior:
```
Initial PIR: ~0.85
Gradual degradation
IDO triggered mid-sequence
Final rating: C / D
```
Unlike the corrupted trajectory (which fails abruptly),
the adversarial case demonstrates **slow causal debt accumulation**.

---

**Why This Matters**

Many modern spatial AI systems (including neural world models and video generators) can produce motion that is:
- **visually coherent**
- **temporally smooth**
- yet **physically inconsistent**

Traditional metrics such as:
- velocity smoothness
- jerk minimization
- pixel consistency

may fail to detect these errors.

SIPA detects them because the **octonion associator exposes non-associative causal drift.**

---

**Interpretation**

The adversarial test illustrates the central claim of the
**Non-Associative Residual Hypothesis (NARH)**:

If a trajectory violates physical causal structure,
algebraic associativity will accumulate residual error over time.

SIPA measures this accumulation as **Physical Debt**, which lowers the **Physical Integrity Rating (PIR)** and triggers **Integrity Degradation Onset (IDO)**.

| Test Type | Visual Smoothness | Physical Validity | SIPA Detection |
|-----------|------------------|------------------|---------------|
| Normal Motion | ✓ | ✓ | Stable PIR |
| Teleportation | ✗ | ✗ | Immediate PIR collapse |
| Adversarial Drift | ✓ | ✗ | Gradual PIR collapse |


---

# 🧠 Core Methodology

**Non-Associative Residual Hypothesis (NARH)**

SIPA's audit engine is built upon the **Non-Associative Residual Hypothesis (NARH)**.

The hypothesis proposes that deviations in causal integrity of a 3D trajectory are reflected in the **associativity behavior of its octonion representation.**

Any "physical hallucination" manifests as a **non-zero octonion associator residual.**

The octonion formulation acts as a **structural consistency probe**, rather than a force-level physics simulator.

Technical references:

[https://github.com/ZC502/Isaac-Sim-Physical-consistency-plugin](https://github.com/ZC502/Isaac-Sim-Physical-consistency-plugin#non-associative-residual-hypothesis-narh)

[https://github.com/ZC502/TinyOEKF/blob/master/docs/Continuous_Physics_Solver_for_AI_Wang_Liu.pdf](https://github.com/ZC502/TinyOEKF/blob/master/docs/Continuous_Physics_Solver_for_AI_Wang_Liu.pdf)

---

# 🧩 What SIPA Can Audit

SIPA is a **trajectory-level physical consistency diagnostic.**

It requires **only motion trajectories** and does not need simulator internals.

Compatible with

**Physics Simulators**
- NVIDIA Isaac Sim
- MuJoCo
- PyBullet
- Gazebo

**Spatial World Models**
- World Labs Marble
- OpenAI Sora (via pose extraction)
- Runway Gen-3

**Robotics Systems**
- Robotic telemetry logs
- Motion capture systems (OptiTrack, Vicon)

---

# 📊 Physical Integrity Rating (PIR)

SIPA introduces the **Physical Integrity Rating (PIR)**
```
PIR = Q_data × (1 − D_phys)
```
Where
- Q_data = data quality
- D_phys = physical debt derived from the octonion residual

**Credit Rating Scale**
| PIR   | Rating | Interpretation          |
| ----- | ------ | ----------------------- |
| ≥0.85 | A      | High physical integrity |
| ≥0.70 | B      | Acceptable              |
| ≥0.50 | C      | Speculative             |
| ≥0.30 | D      | High risk               |
| <0.30 | F      | Critical failure        |

---

# 🔭 Future Directions

Future extensions may include

**Autonomous Driving Forensics**

Applying SIPA-style causal auditing to
- autonomous driving trajectories
- planning stacks
- long-horizon rollouts

Potential targets include systems similar to
- Tesla FSD
- embodied driving world models
- neural simulation stacks

The long-term goal is to build **post-hoc physical sanity checks for safety-critical spatial intelligence systems.**

---

# 💼 Business Value

SIPA transforms qualitative visual plausibility into **quantitative physical solvency.**

Applications include

**Technical Due Diligence**

Quantitatively evaluate world models and simulation claims.

**Sim-to-Real Risk Mitigation**

Detect **kinetic debt** before deploying policies to real robots.

**Automated QA for Smart Spaces**

Monitor physical consistency in digital twins and industrial robotics.

---

# 📢 For the Community

Is that Spatial AI demo **physically real** or just **visually plausible**?

Don't guess.

**Run the Audit.**

---

# ⚖️ Licensing

Academic research use is permitted with attribution.

Commercial deployment requires a separate license agreement.

Patent filing in preparation.

Business inquiries

📧 liuzc19761204@gmail.com

---

# 📚 Citation

If you use SIPA in research please cite

**SIPA: Spatial Intelligence Physical Audit (2026)**
