# ============================================================
# SIPA v1.0 — Video Trajectory Physical Consistency Auditor
# ------------------------------------------------------------
# Engine-agnostic trajectory diagnostics.
#
# INPUT:
#   poses: (T,7) -> [x,y,z,qx,qy,qz,qw]
#
# OUTPUT:
#   - kinematic smoothness
#   - energy proxy drift
#   - octonion associator signal
#   - SNR risk flag
#   - audit plots
# ============================================================

import numpy as np
import json
import os
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def normalize_quat(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / norm


def quat_mul(q1, q2):
    """Hamilton quaternion product"""
    x1, y1, z1, w1 = q1.T
    x2, y2, z2, w2 = q2.T

    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2

    return np.stack([x, y, z, w], axis=1)


# ============================================================
# Core Auditor
# ============================================================

class SIPAVideoAuditor:
    def __init__(self, dt: float):
        self.dt = float(dt)

    # --------------------------------------------------------
    # Main entry
    # --------------------------------------------------------
    def analyze(self, poses: np.ndarray, save_dir="sipa_outputs"):
        """
        poses: (T,7) -> [x,y,z,qx,qy,qz,qw]
        """

        os.makedirs(save_dir, exist_ok=True)

        pos = poses[:, :3]
        quat = normalize_quat(poses[:, 3:7])

        vel = self._finite_diff(pos)
        acc = self._finite_diff(vel)
        jerk = self._finite_diff(acc)

        # --- metrics ---
        smoothness = self._jerk_metric(jerk)
        energy_drift = self._energy_proxy(vel)
        assoc_signal = self._octonion_like_signal(quat)

        snr = self._compute_snr(assoc_signal)

        risk = self._risk_flag(snr)

        report = {
            "smoothness": float(smoothness),
            "energy_proxy_drift": float(energy_drift),
            "octonion_associator_mean": float(np.mean(assoc_signal)),
            "octonion_associator_max": float(np.max(assoc_signal)),
            "snr": float(snr),
            "risk_flag": risk,
        }

        # save report
        with open(os.path.join(save_dir, "sipa_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # plots
        self._plot_signals(
            vel, acc, assoc_signal, save_dir
        )

        return report

    # ========================================================
    # Metrics
    # ========================================================

    def _finite_diff(self, x):
        return np.gradient(x, self.dt, axis=0)

    def _jerk_metric(self, jerk):
        return np.mean(np.linalg.norm(jerk, axis=1))

    def _energy_proxy(self, vel):
        speed2 = np.sum(vel**2, axis=1)
        return float(np.std(speed2))

    # --------------------------------------------------------
    # Octonion-like order sensitivity signal
    # --------------------------------------------------------
    def _octonion_like_signal(self, quat):
        """
        Proxy for non-commutative path sensitivity.

        We compare:
            q_{t+1} * q_t  vs  q_t * q_{t+1}

        Not true octonions (no engine access),
        but a trajectory-level sensitivity probe.
        """

        q1 = quat[:-1]
        q2 = quat[1:]

        forward = quat_mul(q2, q1)
        permuted = quat_mul(q1, q2)

        diff = forward - permuted
        mag = np.linalg.norm(diff, axis=1)

        return mag

    def _compute_snr(self, signal):
        noise_floor = np.median(signal) + 1e-12
        peak = np.max(signal)
        return float(peak / noise_floor)

    def _risk_flag(self, snr):
        if snr < 1.0:
            return "GREEN"
        elif snr < 10.0:
            return "YELLOW"
        else:
            return "RED"

    # ========================================================
    # Visualization
    # ========================================================

    def _plot_signals(self, vel, acc, assoc, save_dir):
        t = np.arange(len(vel))

        plt.figure(figsize=(6, 4))
        plt.plot(t, np.linalg.norm(vel, axis=1), label="|v|")
        plt.plot(t, np.linalg.norm(acc, axis=1), label="|a|")
        plt.legend()
        plt.title("Kinematic Signals")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "kinematics.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(assoc, label="Associator Proxy")
        plt.legend()
        plt.title("Octonion Sensitivity Signal")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "associator.png"), dpi=200)
        plt.close()


# ============================================================
# Quick CLI test
# ============================================================

def _demo():
    T = 300
    dt = 0.02

    t = np.arange(T) * dt

    # synthetic smooth motion
    pos = np.stack([
        np.sin(t),
        np.cos(t),
        0.1 * t
    ], axis=1)

    quat = np.tile(np.array([0, 0, 0, 1]), (T, 1))

    poses = np.concatenate([pos, quat], axis=1)

    auditor = SIPAVideoAuditor(dt)
    report = auditor.analyze(poses)

    print("\n=== SIPA REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    _demo()
