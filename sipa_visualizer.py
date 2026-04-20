import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from core.sipa_yumi_engine import SIPAYuMiEngine, JointSample


# ============================================================
# Helpers
# ============================================================

def extract_joints_from_packet(packet):
    """
    Extract 7-axis IRB 14050 / YuMi-like joints from a RWS-style JSON packet.
    Expected keys:
      rax_1 ... rax_6, eax_a
    Returns degrees.
    """
    res = packet.get("_embedded", {}).get("resources", [{}])[0]

    joints_deg = [
        float(res["rax_1"]),
        float(res["rax_2"]),
        float(res["rax_3"]),
        float(res["rax_4"]),
        float(res["rax_5"]),
        float(res["rax_6"]),
        float(res["eax_a"]),
    ]
    return joints_deg


def format_peak(v: float) -> str:
    """
    Format peak values nicely for annotation.
    Example:
      11224.296 -> 1.1e4
      43.574    -> 43.6
    """
    if not np.isfinite(v) or v <= 0:
        return "0"

    if v < 1000:
        return f"{v:.1f}"

    exp = int(math.floor(math.log10(v)))
    base = v / (10 ** exp)
    return f"{base:.1f}e{exp}"


def find_alert_window(df: pd.DataFrame):
    """
    Find continuous alert window from alarm_flag rows.
    Returns (t_start, t_end) or (None, None)
    """
    alert_rows = df[df["alarm_flag"] == 1]
    if alert_rows.empty:
        return None, None
    return float(alert_rows["time"].min()), float(alert_rows["time"].max())


# ============================================================
# Main visualization
# ============================================================

def run_visualization(
    input_json="debug_payload_seq.json",
    output_png="diagnostic_plot_dual_panel.png",
    dt=0.2,
):
    # --------------------------------------------------------
    # 1. Load replay payloads
    # --------------------------------------------------------
    with open(input_json, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    engine = SIPAYuMiEngine()
    rows = []

    # --------------------------------------------------------
    # 2. Replay packets through the engine
    # --------------------------------------------------------
    for i, packet in enumerate(payloads):
        try:
            joints_deg = extract_joints_from_packet(packet)
        except (KeyError, ValueError, TypeError, IndexError):
            continue

        # Engine expects radians
        joints_rad = np.deg2rad(np.array(joints_deg, dtype=float))

        sample = JointSample(
            q=joints_rad,
            timestamp=i * dt,
            source="debug_replay",
            meta={"raw_unit": "deg"},
        )

        result = engine.update(sample)

        alarm_flag = 0
        alarm_texts = []

        for alarm in result.alarms:
            alarm_texts.append(f"{alarm.severity}:{alarm.kind}={alarm.value:.3f}")
            if str(alarm.severity).upper() == "ALERT":
                alarm_flag = 1

        rows.append(
            {
                "time": i * dt,
                "ready": int(result.ready),
                "assoc_norm": result.associator_norm if result.associator_norm is not None else np.nan,
                "assoc_raw": result.associator_raw if result.associator_raw is not None else np.nan,
                "tcp_step_mm": result.tcp_step_mm if result.tcp_step_mm is not None else np.nan,
                "alarm_flag": alarm_flag,
                "alarm_text": " | ".join(alarm_texts) if alarm_texts else "",
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No valid packets parsed. Nothing to plot.")
        return

    plot_df = df[df["ready"] == 1].copy()

    if plot_df.empty:
        print("Engine never reached ready state. Nothing meaningful to plot.")
        return

    # --------------------------------------------------------
    # 3. Build forum-friendly display score
    # --------------------------------------------------------
    # We use the positive part of associator_norm for display because:
    # - your post talks about 11,000+ peaks
    # - negative normalized values are not visually meaningful in a log plot
    # - this produces a clean "alert score" curve for forum presentation
    plot_df["narh_alert_score"] = plot_df["assoc_norm"].fillna(0.0).clip(lower=0.0)

    # For log plotting we need strictly positive values
    plot_df["narh_plot"] = plot_df["narh_alert_score"].clip(lower=1e-3)

    # Alert window from actual ALERT rows
    t0, t1 = find_alert_window(plot_df)

    # Peak for annotation
    peak_idx = plot_df["narh_alert_score"].idxmax()
    peak_time = float(plot_df.loc[peak_idx, "time"])
    peak_val = float(plot_df.loc[peak_idx, "narh_alert_score"])
    peak_plot_val = float(plot_df.loc[peak_idx, "narh_plot"])

    # --------------------------------------------------------
    # 4. Draw figure
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]},
        constrained_layout=False,
    )

    # ---------------------------
    # Panel A: Estimated TCP step
    # ---------------------------
    ax1.plot(
        plot_df["time"],
        plot_df["tcp_step_mm"],
        color="#1f77b4",
        linewidth=2.8,
        label="Estimated TCP step",
    )

    if t0 is not None and t1 is not None:
        ax1.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            label="Detected discontinuity window",
        )

    ax1.set_title(
        "SIPA Diagnostic: IRB 14050 Redundancy-State Discontinuity",
        fontsize=26,
        pad=14,
    )
    ax1.set_ylabel("TCP Step (mm)", fontsize=22)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", frameon=False, fontsize=16)

    # Tight y-padding so the TCP line is readable
    tcp_min = float(np.nanmin(plot_df["tcp_step_mm"]))
    tcp_max = float(np.nanmax(plot_df["tcp_step_mm"]))
    pad = max((tcp_max - tcp_min) * 0.12, 0.0003)
    ax1.set_ylim(tcp_min - pad, tcp_max + pad)

    # Clean textbox instead of long arrow
    ax1.text(
        0.33,
        0.62,
        "Estimated TCP step remains nearly constant",
        transform=ax1.transAxes,
        fontsize=17,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="gray", alpha=0.92),
    )

    # ---------------------------
    # Panel B: NARH alert score
    # ---------------------------
    ax2.plot(
        plot_df["time"],
        plot_df["narh_plot"],
        color="#d62728",
        linewidth=3.2,
        label="NARH continuity score",
    )
    ax2.set_yscale("log")

    if t0 is not None and t1 is not None:
        ax2.axvspan(t0, t1, color="gray", alpha=0.15)

    ax2.set_xlabel("Time (s)", fontsize=22)
    ax2.set_ylabel("NARH Continuity Score (log scale)", fontsize=20)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper left", frameon=False, fontsize=16)

    # Peak annotation with offset points, not data-space multiplier
    if peak_val > 0:
        ax2.annotate(
            f"Peak: {format_peak(peak_val)}",
            xy=(peak_time, peak_plot_val),
            xytext=(-90, 35),
            textcoords="offset points",
            fontsize=16,
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"),
        )

    # Middle textbox inside detected window
    if t0 is not None and t1 is not None:
        mid_t = 0.5 * (t0 + t1)
        # place the text at a safe log-scale level
        ax2.text(
            mid_t,
            max(1.0, peak_plot_val / 120.0),
            "Hidden redundancy-state discontinuity",
            fontsize=15,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="gray", alpha=0.90),
        )

    # Footer note
    fig.text(
        0.5,
        0.02,
        "Replay benchmark • Read-only RWS-style probe • Not yet plant data",
        ha="center",
        fontsize=16,
        color="dimgray",
    )

    # Tick size
    ax1.tick_params(axis="both", labelsize=16)
    ax2.tick_params(axis="both", labelsize=16)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Success! '{output_png}' has been generated.")

    # Also print a compact summary for sanity-check
    print(f"Peak NARH display score: {peak_val:.3f} at t={peak_time:.3f}s")
    if t0 is not None and t1 is not None:
        print(f"Detected discontinuity window: {t0:.3f}s -> {t1:.3f}s")


if __name__ == "__main__":
    run_visualization()
