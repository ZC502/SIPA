import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Global style (camera-ready, reviewer-safe)
# ============================================================

ENABLE_RISK_BANDS = True
ENABLE_CONFIDENCE_ENVELOPE = True

# Professional palette (not overly saturated)
COLOR_PIR = "#2563EB"          # blue
COLOR_DEBT = "#DC2626"         # red
COLOR_ENVELOPE = "#6B7280"     # neutral gray

# Risk band colors (very light, audit style)
BAND_GREEN = "#ECFDF5"
BAND_YELLOW = "#FFFBEB"
BAND_RED = "#FEF2F2"


# ============================================================
# --- Confidence Envelope (validator-weighted)
# ============================================================

def compute_pir_confidence_envelope(
    pir_curve: np.ndarray,
    validator_score: float,
    window: int = 15,
    k: float = 2.0,
):
    """
    Compute validator-weighted confidence envelope.

    Parameters
    ----------
    pir_curve : (T,)
    validator_score : float in [0,1]
        Higher = better data quality → narrower band
    window : rolling std window
    k : uncertainty amplification factor

    Returns
    -------
    upper, lower : arrays
    """

    if len(pir_curve) < 3:
        return pir_curve, pir_curve

    # rolling std (robust, no pandas dependency)
    pad = window // 2
    padded = np.pad(pir_curve, (pad, pad), mode="edge")

    rolling_std = np.zeros_like(pir_curve)
    for i in range(len(pir_curve)):
        segment = padded[i:i + window]
        rolling_std[i] = np.std(segment)

    # --- validator weighting (KEY DESIGN) ---
    validator_score = float(np.clip(validator_score, 0.0, 1.0))
    uncertainty_scale = 1.0 + k * (1.0 - validator_score)

    envelope_width = rolling_std * uncertainty_scale

    upper = np.clip(pir_curve + envelope_width, 0.0, 1.0)
    lower = np.clip(pir_curve - envelope_width, 0.0, 1.0)

    return upper, lower


# ============================================================
# --- Risk Bands (Diagnostic Risk Thresholding)
# ============================================================

def _draw_risk_bands(ax):
    """Draw Green / Yellow / Red background bands."""

    # Green zone
    ax.axhspan(0.7, 1.0, color=BAND_GREEN, alpha=0.6, zorder=0)

    # Yellow zone
    ax.axhspan(0.4, 0.7, color=BAND_YELLOW, alpha=0.6, zorder=0)

    # Red zone
    ax.axhspan(0.0, 0.4, color=BAND_RED, alpha=0.6, zorder=0)


# ============================================================
# --- Main Plot Function (Production)
# ============================================================

def plot_pir_evolution(
    pir_t,
    debt_t,
    dt,
    onset_info,
    save_dir,
    prefix="sipa_audit",
    validator_score=0.95,
):
    """
    Generate SIPA Physical Integrity evolution figure.

    Features
    --------
    ✓ Diagnostic Risk Thresholding (background bands)
    ✓ PIR curve
    ✓ Physical Debt curve
    ✓ Validator-weighted confidence envelope
    ✓ Integrity onset marker
    ✓ PNG + PDF export
    """

    pir_t = np.asarray(pir_t, dtype=float)
    debt_t = np.asarray(debt_t, dtype=float)

    t = np.arange(len(pir_t)) * dt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Figure setup (ICRA/IEEE safe)
    # =========================================================
    plt.figure(figsize=(6.0, 4.0))
    ax = plt.gca()

    # =========================================================
    # Background risk bands
    # =========================================================
    if ENABLE_RISK_BANDS:
        _draw_risk_bands(ax)

    # =========================================================
    # Confidence envelope (MIDDLE LAYER)
    # =========================================================
    if ENABLE_CONFIDENCE_ENVELOPE and len(pir_t) > 5:
        upper, lower = compute_pir_confidence_envelope(
            pir_curve=pir_t,
            validator_score=validator_score,
        )

        ax.fill_between(
            t,
            lower,
            upper,
            color=COLOR_ENVELOPE,
            alpha=0.22,
            linewidth=0,
            label="Confidence Envelope",
            zorder=2,
        )

    # =========================================================
    # PIR curve (TOP LAYER)
    # =========================================================
    ax.plot(
        t,
        pir_t,
        color=COLOR_PIR,
        linewidth=2.2,
        label="PIR(t)",
        zorder=3,
    )

    # =========================================================
    # Physical debt curve
    # =========================================================
    if len(debt_t) == len(pir_t):
        ax.plot(
            t,
            debt_t,
            linestyle="--",
            color=COLOR_DEBT,
            linewidth=1.6,
            alpha=0.9,
            label="Physical Debt(t)",
            zorder=3,
        )

    # =========================================================
    # Risk threshold line
    # =========================================================
    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.6,
        label="Risk Threshold",
        zorder=1,
    )

    # =========================================================
    # Onset marker
    # =========================================================
    if onset_info is not None and "time_sec" in onset_info:
        ax.axvline(
            onset_info["time_sec"],
            linestyle="-.",
            linewidth=1.6,
            color="black",
            alpha=0.8,
            label="Integrity Degradation Onset",
            zorder=4,
        )

    # =========================================================
    # Labels & styling
    # =========================================================
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title("SIPA Physical Integrity Evolution")

    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.02)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    # =========================================================
    # Save outputs
    # =========================================================
    png_path = save_dir / f"{prefix}_pir_evolution.png"
    pdf_path = save_dir / f"{prefix}_pir_evolution.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.savefig(pdf_path)
    plt.close()

    return str(png_path), str(pdf_path)
