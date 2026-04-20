import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from core.sipa_yumi_engine import SIPAYuMiEngine, JointSample


def extract_joints_from_packet(packet):
    """
    Extract 7-axis YuMi/IRB14050-like joints from a RWS-style JSON packet.
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


def find_alert_window(df):
    """
    Find a continuous alert window from alarm flags.
    Returns (t_start, t_end) or (None, None).
    """
    alert_rows = df[df["alarm_flag"] == 1]
    if alert_rows.empty:
        return None, None
    return alert_rows["time"].min(), alert_rows["time"].max()


def format_peak(v):
    """
    Format large numbers for annotation.
    Example: 1513712511 -> 1.5e9
    """
    if v <= 0:
        return "0"
    exp = int(math.floor(math.log10(v)))
    base = v / (10 ** exp)
    return f"{base:.1f}e{exp}"


def run_visualization(
    input_json="debug_payload_seq.json",
    output_png="diagnostic_plot_dual_panel.png",
    dt=0.2,
):
    # --------------------------------------------------------
    # 1. Load payload sequence
    # --------------------------------------------------------
    with open(input_json, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    engine = SIPAYuMiEngine()
    rows = []

    # --------------------------------------------------------
    # 2. Replay packets through engine
    # --------------------------------------------------------
    for i, packet in enumerate(payloads):
        try:
            joints_deg = extract_joints_from_packet(packet)
        except (KeyError, ValueError, TypeError):
            continue

        # IMPORTANT: engine expects radians
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
        for a in result.alarms:
            alarm_texts.append(f"{a.severity}:{a.kind}={a.value:.3f}")
            if a.severity.upper() == "ALERT":
                alarm_flag = 1

        rows.append(
            {
                "time": i * dt,
                "ready": int(result.ready),
                "assoc_raw": result.associator_raw if result.associator_raw is not None else np.nan,
                "assoc_norm": result.associator_norm if result.associator_norm is not None else np.nan,
                "tcp_step_mm": result.tcp_step_mm if result.tcp_step_mm is not None else np.nan,
                "alarm_flag": alarm_flag,
                "alarm_text": " | ".join(alarm_texts) if alarm_texts else "",
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No valid packets parsed. Nothing to plot.")
        return

    # --------------------------------------------------------
    # 3. Plot prep
    # --------------------------------------------------------
    t0, t1 = find_alert_window(df)

    plot_df = df[df["ready"] == 1].copy()
    if plot_df.empty:
        print("Engine never reached ready state. Nothing meaningful to plot.")
        return

    # For log plotting: use raw positive score, not normalized score
    plot_df["assoc_plot"] = np.log10(plot_df["assoc_raw"].fillna(np.nan).clip(lower=1e-9))

    peak_idx = plot_df["assoc_plot"].idxmax()
    peak_time = plot_df.loc[peak_idx, "time"]
    peak_val = plot_df.loc[peak_idx, "assoc_plot"]

    # --------------------------------------------------------
    # 4. Draw dual-panel figure
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3]},
    )

    # ---------------------------
    # Panel A: Estimated TCP step
    # ---------------------------
    ax1.plot(
        plot_df["time"],
        plot_df["tcp_step_mm"],
        color="#1f77b4",
        linewidth=2.2,
        label="Estimated TCP step",
    )

    if t0 is not None and t1 is not None:
        ax1.axvspan(t0, t1, color="gray", alpha=0.15, label="Detected discontinuity window")

    ax1.set_title("SIPA Diagnostic: IRB 14050 Redundancy-State Discontinuity", fontsize=18, pad=10)
    ax1.set_ylabel("TCP Step (mm)", fontsize=13)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", frameon=False)

    # text annotation
    tcp_mean = np.nanmean(plot_df["tcp_step_mm"])
    ax1.annotate(
        "Estimated TCP step remains nearly constant",
        xy=(plot_df["time"].iloc[len(plot_df)//2], tcp_mean),
        xytext=(plot_df["time"].min() + 0.1, tcp_mean + 0.12),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        fontsize=11,
    )

    # ---------------------------
    # Panel B: NARH continuity
    # ---------------------------
    ax2.plot(
        plot_df["time"],
        plot_df["assoc_plot"],
        color="#d62728",
        linewidth=2.6,
        label="NARH continuity score",
    )

    ax2.set_ylabel("log10(NARH Continuity Score)", fontsize=13)

    if t0 is not None and t1 is not None:
        ax2.axvspan(t0, t1, color="gray", alpha=0.15)

    ax2.annotate(
        f"Peak: {format_peak(peak_val)}",
        xy=(peak_time, peak_val),
        ax2.annotate(
    f"Peak: {format_peak(raw_peak_val)}",
    xy=(peak_time, peak_display_val),
    xytext=(20, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
    fontsize=11,
),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
        fontsize=11,
    )

    if t0 is not None and t1 is not None:
        mid_t = 0.5 * (t0 + t1)
        ax2.text(
            mid_t,
            peak_val / 30,
            "Hidden redundancy-state discontinuity",
            fontsize=11,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )

    ax2.set_xlabel("Time (s)", fontsize=13)
    ax2.set_ylabel("NARH Continuity Score (log scale)", fontsize=13)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)

    # Footer note
    fig.text(
        0.5,
        0.01,
        "Replay benchmark • Read-only RWS-style probe • Not yet plant data",
        ha="center",
        fontsize=10,
        color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Success! '{output_png}' has been generated.")


if __name__ == "__main__":
    run_visualization()
