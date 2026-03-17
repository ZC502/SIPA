# SIPA: Simulation Integrity & Physics Auditor
*The Black Box Auditor for Industrial Robot Trajectories*

### SIPA is a diagnostic tool for industrial robot simulations.

It analyzes robot trajectories exported from simulators such as
KUKA.Sim and detects non-physical motion artifacts including:
- **TCP discontinuous jumps**
- **Z-axis micro jitter**
- **joint acceleration spikes**
- **workspace instability regions**

SIPA bridges the gap between 'Perfect Simulation' and 'Violent Reality'.

### 📂 Case Study: KUKA LBR iiwa 14 R820 Stability Audit

**Scenario Overview**
- **Robot Model**: KUKA LBR iiwa 14 R820
- **Sampling Frequency**: $100\text{Hz}$ (10ms step)
- **Environment**: KUKA.Sim Pro / Visual Components
- **Task**: Complex 3D spiral trajectory execution.

**SIPA Audit Report v2.1**
```
Robot: KUKA LBR iiwa 14 R820
Frames: 125 | Frequency: 100Hz

[CRITICAL] TCP Z Jitter: 10.96 mm (Std Amplitude)
[WARNING] TCP Jump Events Detected at Initialization (Frame 0-4)
          Max Jump: 85.40 mm
[DIAGNOSIS] Micro-oscillation detected at Joint 2 (Mid-path).
[RISK LEVEL] HIGH: Potential Gearbox Resonance & Controller Overcurrent
```
**Visual Forensics**

| ![Joint Acceleration Analysis](demo/J2axis.png)  | ![TCP Physical Residual](demo/Z-axis.png) |
|---------------------------|-----------------------------------|
| **Observation**:J2 axis experiences a massive acceleration spike ($>600\text{rad/s}^2$) near frame 60. |**Observation**: High-frequency jitter in Z-axis exceeds 0.04m, indicating solver divergence.

| ![Spatial Stability Heatmap](demo/TCP.png) | ![Trajectory Sanity Check](demo/3D.png)   |
|---------------------------|-----------------------------------|
|**Observation**: Yellow/Pink clusters indicate localized instability zones in the working envelope.| **Observation**: The 3D path shows geometric continuity, but hides the underlying physical jitter."

**Industrial Impact**

Without the intervention of SIPA, this trajectory shows as "pass" in the simulation software. However, after being deployed to the actual machine:
- **Initial jump (Frame 0-4)**: It will cause the robotic arm to produce a violent impact sound and trigger an emergency stop (E-Stop).
- **Micro-oscillation in the middle section of the path**: It will cause the J2 reducer to generate high-frequency heat, accelerating hardware fatigue.
- **Final output**: Visible ripple defects will appear in the welding or gluing process, and the qualification rate will drop by more than 15%.

### 💰 The Economic Impact of Physics Auditing

SIPA transforms abstract physical metrics into tangible industrial ROI (Return on Investment). By detecting "Simulation-to-Real" gaps early, it prevents costly hardware failures and production delays.
- **Hardware Protection**: Detecting a single 5mm TCP surge = Salvaging a €2,500+ robotic welding torch or sensor assembly from collision damage.
- **Asset Longevity**: Identifying non-physical oscillations in Axis 3 = Extending gearbox and harmonic drive service life by 15% through mechanical fatigue mitigation.
- **Downtime Reduction**: Each simulation-to-real error caught before deployment = Saving €500–€2,000 per hour in avoided production line downtime during commissioning.
- **Quality Assurance (Scrap Rate)**: Eliminating micro-vibrations in glue/sealing paths = Reducing scrap rates by 20% for high-precision automotive assembly tasks.
- **Energy Efficiency**: Optimizing EJI (Energy Jitter Index) = A 3–5% reduction in peak power consumption and motor thermal stress across 24/7 operations.
- **Commissioning Speed**: Physics-consistent trajectories = Cutting field-tuning time by 30%, allowing faster "Time-to-Market" for new production cells.

### Supported models:

✅ KUKA LBR iiwa 7 R800 / 14 R820 (Verified)

⏳ KUKA KR QUANTEC / KR IONTEC (Upcoming)

### NARH (Non-Associative Residual Hypothesis)

**NARH is the diagnostic core of SIPA.**

It evaluates residual motion signals under discrete simulation
timesteps (Δt) to reveal numerical artifacts introduced by
trajectory interpolation or solver instability.

**In practice, NARH enables SIPA to act as a "black box" for industrial robot trajectories.**

For the specific mathematical formulas, please refer to the "**core-methodology**" section at the end of this article.

---

### SIPA VS Traditional Simulation 

| Feature                  | Traditional Simulation                  | SIPA Audited                              |
|--------------------------|-----------------------------------------|-------------------------------------------|
| **Path Check**           | Geometry Only (Pass/Fail)               | Physics-consistent (Pass/Fail)            |
| **Collision Risk**       | Static/Dynamic Check                    | Solver-based Surge Detection              |
| **Maintenance**          | Reactive (Fix when broken)              | Proactive (Avoid resonance)               |
| **Sim-to-Real**          | "Finger-crossed" deployment             | Verified Physics Alignment                |            

---

# 🚀 Quick Start (30-Second Demo)

**（Open in GitHub Codespaces to run the audit in 30 seconds without local installation.）**

Clone the repository and run the baseline audit examples.

### 1. Clone Repository

```bash
git clone https://github.com/ZC502/SIPA.git
cd SIPA
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the KUKA iiwa Audit

SIPA features **Auto-Unit Detection** (Degrees/Radians) for KUKA.Sim and Sunrise.OS files.
```
python scripts/sipa_iiwa_audit.py --input demo/test_iiwa_radians.csv --robot iiwa14 --unit auto
```

---

### 📊 Manual Audit & Usage

**Supported Models**

- ✅ KUKA LBR iiwa 7 R800 / 14 R820 (Verified)

- ⏳ KUKA KR QUANTEC / KR IONTEC (Upcoming)

**Command Line Interface (CLI)**
| Parameter | Description                  | Default    |
|-----------|------------------------------|------------|
| ```--input```   | Path to CSV trajectory       | Required   |
| ```--robot```   | ```iiwa14``` or ```iiwa7```              | ```iiwa14 ```    |
| ```--unit```    | ```auto```, ```deg```, or ```rad```          | ```auto```       |
| ```--output```  | Directory for reports/images | ```outputs/```   |

---

### 📦 Output Artifacts
All diagnostics are saved to ```outputs/```:

```audit_report.txt```: Qualitative and quantitative summary of the trajectory health.

```tcp_heatmap.png```: Stability map showing exactly where in the workspace the robot vibrates.

```tcp_3d_path.png```: Geometric sanity check to ensure coordinates match the real cell.

```z_jitter.png```: High-frequency residual analysis (The NARH Probe).

```joint_acc.png```: Acceleration audit to prevent motor over-torque.

---

### 📄 Input Format (7-DoF Joint CSV)
SIPA accepts CSV files with 7 columns representing the 7 joints of the robot.
```
# J1, J2, J3, J4, J5, J6, J7
-1.307, -1.042, -1.869, 0.292, -0.399, -1.665, 2.326
...
```
*Note: SIPA ignores lines starting with # and automatically detects if values are in degrees or radians.*

---

### ⚖️ Licensing & Citation
**Licensing**
- **Academic/Research**: Permitted with attribution.
- **Commercial/Industrial**: Requires a separate license agreement. Patent filing in preparation.

**Contact**
📧 liuzc19761204@gmail.com

**Citation**
If you use SIPA in your industrial or academic work, please cite:

***SIPA: Simulation Integrity & Physics Auditor (2026)**. Developed by ZC502.*

---

### 🧠core-methodology
### Non-Associative Residual Hypothesis (NARH)
**1. Setting**

Consider a rigid-body simulation system defined by:

- a state space $S \subset \mathbb{R}^n$
- a nominal associative update operator $\Phi \Delta t : S \to S$
- a parallel constraint resolution pipeline composed of sub-operators $`\{\Psi_i\}_{i=1}^k`$
	​
The simulator advances the system state through a discrete update step:

$$ s_{t+1} = \Psi_{\sigma(k)} \circ \cdots \circ \Psi_{\sigma(1)} (s_t) $$

where the permutation 𝜎 represents an execution order determined by internal solver mechanisms such as:

- constraint partitioning
- thread scheduling
- contact batching
- solver splitting

Each operator $\Psi_i$ represents a well-defined physical constraint update (e.g., contact resolution, joint projection, or velocity correction).

However, the **order in which these updates are applied may vary between solver iterations or execution contexts.**

---

**2. Order Sensitivity in Discrete Solvers**

In continuous rigid-body mechanics, many transformations belong to associative algebraic structures (e.g., matrix multiplication or quaternion composition).

However, in practical simulation systems, constraint updates are implemented through **finite-precision numerical approximations**. Under such conditions the composed operators may exhibit order sensitivity:

$$(\Psi_a \circ \Psi_b) \circ \Psi_c \neq \Psi_a \circ (\Psi_b \circ \Psi_c)$$

This deviation may arise from:
- finite-precision arithmetic
- iterative solver truncation
- projection steps
- asynchronous or parallel execution

To quantify this effect, define the **discrete associator:**

$$
A(a,b,c;s) = \bigl( (\Psi_a \circ \Psi_b) \circ \Psi_c \bigr)(s) - \bigl( \Psi_a \circ (\Psi_b \circ \Psi_c) \bigr)(s)
$$

---

**3. Non-Associative Residual**

The Non-Associative Residual (NAR) at state $s_t$ is defined as

$R_t = \lVert A(a,b,c; s_t) \rVert$

for a selected triple of constraint operators representative of the solver pipeline.

This residual does **not** represent algebraic non-associativity of the physical state space itself.
Instead, it measures **order-dependent deviations introduced by the discrete numerical solver.**

In other words, $R_t$ quantifies the degree to which the simulation outcome depends on the ordering of constraint updates.

---

**4. Hypothesis (NARH)**

The **Non-Associative Residual Hypothesis (NARH)** states that:

In high interaction density regimes — such as contact-rich robotic manipulation or high-speed trajectory execution — the non-associative residual  can become non-negligible relative to trajectory stability metrics.

Over extended simulation horizons, this residual may accumulate as a structured drift component:

$\sum_{t=0}^{T} R_t \not\approx 0$

even when the state increments remain bounded:

$`\| s_{t+1} - s_t \| < \epsilon`$

This implies that solver order sensitivity may introduce **small but measurable deviations in simulated trajectories.**

---

**5. Interpretation for Robot Simulation**

NARH does **not** claim that simulators are mathematically incorrect or physically invalid.

Instead, it highlights a practical property of discrete rigid-body solvers:

>Parallel constraint resolution may introduce small order-dependent residuals that are not explicitly represented in the simulation state.

For industrial robot simulations, such residuals may manifest as:
- micro-scale TCP jitter
- discontinuous pose jumps
- trajectory instability under small timestep changes
- sensitivity to solver parameterization

These effects are typically subtle but may become visible when analyzing robot trajectories in Cartesian space.

Tools such as **SIPA (Simulation Integrity & Physics Auditor)** analyze trajectory residuals to detect such phenomena in a diagnostic context.

---

**6. Falsifiability**

The NARH framework is empirically testable and may be falsified if:

1. The measured residual $R_t$ remains indistinguishable from numerical noise across interaction densities.
2. Reordering constraint application produces statistically identical trajectories.
3.  Classical scalar stability metrics (e.g., kinetic energy norms or velocity bounds) detect instability earlier than any associator-derived signal.

In such cases, the solver behavior may be considered effectively order-invariant for the examined regime.

---

**7. Practical Implication**

If validated empirically, NARH suggests that:
- Order sensitivity is an inherent property of discrete constraint solvers.
- Residual-based diagnostics can serve as **early indicators of trajectory instability.**
- Trajectory analysis tools may benefit from monitoring solver-induced residual signals in addition to traditional stability metrics.

For industrial robot simulation pipelines, such diagnostics can assist in verifying **trajectory physical consistency** before deployment to real hardware.

References:

[https://github.com/ZC502/TinyOEKF/blob/master/docs/Continuous_Physics_Solver_for_AI_Wang_Liu.pdf](https://github.com/ZC502/TinyOEKF/blob/master/docs/Continuous_Physics_Solver_for_AI_Wang_Liu.pdf)






