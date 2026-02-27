import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Global style (camera-ready, reviewer-safe)
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.0,
})

ENABLE_RISK_BANDS = True
ENABLE_CONFIDENCE_ENVELOPE = True

COLOR_PIR = "#2563EB"
COLOR_DEBT = "#DC2626"
COLOR_ENVELOPE = "#6B7280"

BAND_GREEN = "#ECFDF5"
BAND_YELLOW = "#FFFBEB"
BAND_RED = "#FEF2F2"


# ============================================================
# 🔍 RED Zone Detection Logic
# ============================================================

def find_first_red_entry(
    pir: np.ndarray,
    validator_mask: np.ndarray | None = None,
    red_threshold: float = 0.5,
    patience: int = 3,
):
    if pir is None or len(pir) == 0:
        return None

    pir = np.asarray(pir, dtype=float)

    if validator_mask is None:
        validator_mask = np.ones_like(pir, dtype=bool)
    else:
        validator_mask = np.asarray(validator_mask).astype(bool)

    above = (pir >= red_threshold) & validator_mask

    run = 0
    for i, flag in enumerate(above):
        if flag:
            run += 1
            if run >= patience:
                return i - patience + 1
        else:
            run = 0

    return None


def overlay_first_red_marker(
    ax,
    t,
    pir,
    validator_mask=None,
    red_threshold: float = 0.5,
    patience: int = 3,
    zorder: int = 6,
):
    idx = find_first_red_entry(
        pir,
        validator_mask=validator_mask,
        red_threshold=red_threshold,
        patience=patience,
    )

    if idx is None:
        return None

    x = t[idx]
    y = pir[idx]

    # main marker
    ax.scatter(
        [x],
        [y],
        s=90,
        color="#ff2b2b",
        edgecolors="white",
        linewidths=1.5,
        zorder=zorder,
        label="First RED entry",
    )

    # glow
    ax.scatter(
        [x],
        [y],
        s=220,
        color="#ff2b2b",
        alpha=0.18,
        linewidths=0,
        zorder=zorder - 1,
    )

    # guide line
    ax.axvline(
        x,
        color="#ff2b2b",
        alpha=0.18,
        linewidth=1.2,
        linestyle="--",
        zorder=zorder - 2,
    )

    return idx


# ============================================================
# Confidence Envelope
# ============================================================

def compute_pir_confidence_envelope(
    pir_curve: np.ndarray,
    validator_score: float,
    window: int = 15,
    k: float = 2.0,
):
    pir_curve = np.asarray(pir_curve, dtype=float)

    if len(pir_curve) < 3:
        return pir_curve, pir_curve

    pad = window // 2
    padded = np.pad(pir_curve, (pad, pad), mode="edge")

    rolling_std = np.zeros(len(pir_curve), dtype=float)
    for i in range(len(pir_curve)):
        segment = padded[i:i + window]
        rolling_std[i] = np.std(segment)

    validator_score = float(np.clip(validator_score, 0.0, 1.0))
    uncertainty_scale = 1.0 + k * (1.0 - validator_score)

    envelope_width = rolling_std * uncertainty_scale

    upper = np.clip(pir_curve + envelope_width, 0.0, 1.0)
    lower = np.clip(pir_curve - envelope_width, 0.0, 1.0)

    return upper, lower


# ============================================================
# Risk Bands
# ============================================================

def _draw_risk_bands(ax):
    ax.axhspan(0.7, 1.0, color=BAND_GREEN, alpha=0.6, zorder=0)
    ax.axhspan(0.4, 0.7, color=BAND_YELLOW, alpha=0.6, zorder=0)
    ax.axhspan(0.0, 0.4, color=BAND_RED, alpha=0.6, zorder=0)


# ============================================================
# Main Plot
# ============================================================

def plot_pir_evolution(
    pir_t,
    debt_t,
    dt,
    onset_info,
    save_dir,
    prefix="sipa_audit",
    validator_score=0.95,
    validator_mask=None,
):
    pir_t = np.asarray(pir_t, dtype=float)
    debt_t = np.asarray(debt_t, dtype=float)

    t = np.arange(len(pir_t)) * dt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.0, 4.0))
    ax = plt.gca()

    # background bands
    if ENABLE_RISK_BANDS:
        _draw_risk_bands(ax)

    # confidence envelope
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

    # PIR curve
    ax.plot(
        t,
        pir_t,
        color=COLOR_PIR,
        linewidth=2.2,
        label="PIR(t)",
        zorder=3,
    )

    # Physical debt
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

    # threshold
    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.6,
        label="Risk Threshold",
        zorder=1,
    )

    # onset line
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

    # 🔥 CRITICAL: RED marker overlay
    overlay_first_red_marker(
        ax,
        t,
        pir_t,
        validator_mask=validator_mask,
        red_threshold=0.5,
        patience=3,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title("SIPA Physical Integrity Evolution")

    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.02)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    png_path = save_dir / f"{prefix}_pir_evolution.png"
    pdf_path = save_dir / f"{prefix}_pir_evolution.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.savefig(pdf_path)
    plt.close()

    return str(png_path), str(pdf_path)
