import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_pir_evolution(
    pir_t,
    debt_t,
    dt,
    onset_info,
    save_dir,
    prefix="sipa_audit",
    enable_drt=True,  # 🔥 Diagnostic Risk Thresholding (default ON)
):
    """
    Generate SIPA Physical Integrity evolution figure.

    Features:
    - PIR(t) curve
    - Physical Debt(t) curve
    - Risk threshold line
    - Integrity degradation onset marker
    - 🔥 Diagnostic Risk Thresholding (background bands)

    Returns:
        (png_path, pdf_path)
    """

    # ===============================
    # 🔒 Input validation
    # ===============================
    if pir_t is None or len(pir_t) == 0:
        raise ValueError("pir_t is empty")

    if debt_t is None or len(debt_t) == 0:
        raise ValueError("debt_t is empty")

    if dt <= 0:
        raise ValueError("dt must be positive")

    # length alignment (robust against mismatch)
    n = min(len(pir_t), len(debt_t))
    pir_t = np.asarray(pir_t[:n])
    debt_t = np.asarray(debt_t[:n])

    t = np.arange(n) * dt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 🎨 Figure
    # ===============================
    plt.figure(figsize=(7, 4.5))

    # ===============================
    # 🔥 Diagnostic Risk Thresholding
    # ===============================
    if enable_drt:
        y_min = min(np.min(pir_t), np.min(debt_t))
        y_max = max(np.max(pir_t), np.max(debt_t))

        # Green band (stable)
        plt.axhspan(0.75, y_max, alpha=0.08, color="green", label="_nolegend_")

        # Yellow band (marginal)
        plt.axhspan(0.5, 0.75, alpha=0.08, color="gold", label="_nolegend_")

        # Red band (risk)
        plt.axhspan(y_min, 0.5, alpha=0.08, color="red", label="_nolegend_")

    # ===============================
    # 📈 Curves
    # ===============================
    plt.plot(t, pir_t, linewidth=2.2, label="PIR(t)")
    plt.plot(
        t,
        debt_t,
        linestyle="--",
        linewidth=1.8,
        label="Physical Debt(t)",
    )

    # ===============================
    # 🚨 Risk threshold
    # ===============================
    plt.axhline(
        0.5,
        linestyle=":",
        linewidth=1.5,
        label="Risk Threshold",
    )

    # ===============================
    # 📍 Onset marker (robust)
    # ===============================
    if onset_info is not None and isinstance(onset_info, dict):
        onset_time = onset_info.get("time_sec", None)
        if onset_time is not None:
            plt.axvline(
                onset_time,
                linestyle="-.",
                linewidth=1.5,
                label="Integrity Degradation Onset",
            )

    # ===============================
    # 🧾 Labels
    # ===============================
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.title("SIPA Physical Integrity Evolution")

    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)

    # ===============================
    # 💾 Save
    # ===============================
    png_path = save_dir / f"{prefix}_pir_evolution.png"
    pdf_path = save_dir / f"{prefix}_pir_evolution.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.savefig(pdf_path)
    plt.close()

    return str(png_path), str(pdf_path)
