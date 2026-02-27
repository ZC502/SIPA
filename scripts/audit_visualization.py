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
):
    """
    Generate SIPA time-evolution figure.

    Parameters
    ----------
    pir_t : array-like, shape (T,)
        PIR time series in [0, 1]
    debt_t : array-like, shape (T,)
        Physical debt time series in [0, 1]
    dt : float
        Timestep in seconds
    onset_info : dict or None
        Output of detect_integrity_onset()
    save_dir : str or Path
        Output directory
    prefix : str
        Filename prefix

    Returns
    -------
    png_path : str
    pdf_path : str
    num_frames : int
    """

    # -------------------------------
    # Defensive checks (reviewer-safe)
    # -------------------------------
    pir_t = np.asarray(pir_t, dtype=float)
    debt_t = np.asarray(debt_t, dtype=float)

    if pir_t.size == 0:
        raise ValueError("[SIPA] pir_t is empty.")

    if pir_t.shape != debt_t.shape:
        raise ValueError(
            f"[SIPA] Shape mismatch: pir_t {pir_t.shape} vs debt_t {debt_t.shape}"
        )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"[SIPA] Invalid dt: {dt}")

    # -------------------------------
    # Time axis
    # -------------------------------
    t = np.arange(len(pir_t)) * float(dt)

    # -------------------------------
    # Prepare output dir
    # -------------------------------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(6, 4))

    # PIR curve
    plt.plot(t, pir_t, label="PIR(t)")

    # Debt curve
    plt.plot(t, debt_t, linestyle="--", label="Physical Debt(t)")

    # Risk threshold
    plt.axhline(0.5, linestyle=":", label="Risk Threshold")

    # Onset marker (robust)
    if isinstance(onset_info, dict):
        onset_time = onset_info.get("time_sec", None)
        if onset_time is not None and np.isfinite(onset_time):
            if 0 <= onset_time <= t[-1] + 1e-9:
                plt.axvline(
                    onset_time,
                    linestyle="-.",
                    label="Integrity Degradation Onset",
                )

    # Labels
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.title("SIPA Physical Integrity Evolution")

    # 🔑 Reviewer-friendly fixed scale
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.grid(True, alpha=0.3)

    # -------------------------------
    # Save
    # -------------------------------
    png_path = save_dir / f"{prefix}_pir_evolution.png"
    pdf_path = save_dir / f"{prefix}_pir_evolution.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    return str(png_path), str(pdf_path), int(len(pir_t))
