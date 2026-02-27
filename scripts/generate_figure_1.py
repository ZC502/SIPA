#!/usr/bin/env python3
"""
ICRA Figure 1 Generator
SIPA v1.0

Generates the Green / Yellow / Red audit visualization
with dual-engine placeholder curves.

Outputs:
    - figure_1_audit.png
    - figure_1_audit.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def generate_placeholder_curves(x):
    """
    Create representative placeholder curves.
    These are NOT real engine data.
    """

    # Isaac-like: steeper instability
    isaac = 0.8 * x**1.6

    # Marble-like: more stable regime
    marble = 0.5 * x**1.2

    return isaac, marble


def create_figure(save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Domain
    # -------------------------
    x = np.linspace(0.1, 10, 400)
    isaac, marble = generate_placeholder_curves(x)

    # -------------------------
    # Figure setup (ICRA style)
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # -------------------------
    # Zones
    # -------------------------
    plt.axhspan(0, 1, alpha=0.15, label="Green Zone (Associative)")
    plt.axhspan(1, 3, alpha=0.15, label="Yellow Zone (Warning)")
    plt.axhspan(3, 12, alpha=0.15, label="Red Zone (Collapse)")

    # -------------------------
    # Curves
    # -------------------------
    plt.plot(x, isaac, linewidth=2, label="Engine A (e.g., Isaac-like)")
    plt.plot(x, marble, linewidth=2, linestyle="--",
             label="Engine B (e.g., Marble-like)")

    # Threshold lines
    plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.axhline(3.0, linestyle=":", linewidth=1)

    # -------------------------
    # Labels
    # -------------------------
    plt.xlabel("Scene Density / Constraint Load")
    plt.ylabel("Causal Signal-to-Noise Ratio (SNR)")
    plt.title("Figure 1: Auditable Physical Consistency Regimes")

    plt.xlim(0, 10)
    plt.ylim(0, 12)

    plt.legend(frameon=False)
    plt.grid(alpha=0.25)

    plt.tight_layout()

    # -------------------------
    # Export
    # -------------------------
    png_path = os.path.join(save_dir, "figure_1_audit.png")
    pdf_path = os.path.join(save_dir, "figure_1_audit.pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"[SIPA] Figure saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    create_figure()
