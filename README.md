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
