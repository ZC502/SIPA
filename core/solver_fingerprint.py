"""
SIPA v2.2
Solver Instability Fingerprint

Detects numerical solver divergence events in industrial robot simulations.

Part of:
SIPA (Simulation Integrity & Physics Auditor)

Core theory:
NARH (Non-Associative Residual Hypothesis)
"""

import numpy as np
import pandas as pd


class SolverInstabilityFingerprint:

    def __init__(
        self,
        tcp_threshold_mm=50,
        acc_threshold=500,
        z_threshold_mm=50
    ):
        """
        Parameters

        tcp_threshold_mm :
            TCP position jump threshold

        acc_threshold :
            joint acceleration spike (rad/s²)

        z_threshold_mm :
            vertical residual spike
        """

        self.tcp_threshold = tcp_threshold_mm
        self.acc_threshold = acc_threshold
        self.z_threshold = z_threshold_mm

    def detect(
        self,
        tcp_positions,
        joint_acc,
        z_residual
    ):
        """
        Detect solver instability frames.

        Parameters
        ----------
        tcp_positions : Nx3 array (mm)
        joint_acc : Nx7 array (rad/s^2)
        z_residual : N array (mm)

        Returns
        -------
        list of instability events
        """

        tcp_positions = np.array(tcp_positions)
        joint_acc = np.array(joint_acc)
        z_residual = np.array(z_residual)

        events = []

        for i in range(2, len(tcp_positions)):

            tcp_jump = np.linalg.norm(
                tcp_positions[i] - tcp_positions[i-1]
            )

            acc_spike = np.max(np.abs(joint_acc[i]))

            z_spike = abs(z_residual[i])

            if (
                tcp_jump > self.tcp_threshold
                and acc_spike > self.acc_threshold
                and z_spike > self.z_threshold
            ):

                events.append({
                    "frame": i,
                    "tcp_spike_mm": round(tcp_jump,2),
                    "joint_acc_spike": round(acc_spike,2),
                    "z_residual_mm": round(z_spike,2)
                })

        return events


def print_instability_report(events):

    if len(events) == 0:
        print("\n✔ No solver instability detected.")
        return

    print("\n===============================")
    print("Solver Instability Fingerprint")
    print("===============================")

    for e in events:

        print(f"\nFrame {e['frame']}")
        print("Detected solver divergence signature")

        print(f"TCP spike: {e['tcp_spike_mm']} mm")
        print(f"Joint acceleration spike: {e['joint_acc_spike']} rad/s²")
        print(f"Z residual spike: {e['z_residual_mm']} mm")

        print("→ instability evidence")
