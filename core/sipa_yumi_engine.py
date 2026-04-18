#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sipa_yumi_engine.py

SIPA for ABB YuMi
Unified audit core for:
1) Offline CSV batch analysis
2) Live RWS streaming probe (alarm-only)

Design notes
------------
- One audit engine, two input modes.
- Adapters (CSV / RWS) provide JointSample objects.
- The engine owns the rolling-state, scoring, alarming, and optional nominal FK.

Important v1 choices
--------------------
- Joint-space NARH / associator is the primary evidence layer.
- Nominal FK is auxiliary only until a proper YuMi-specific FK model is injected.
- For live RWS mode, history buffers are bounded to avoid unbounded memory growth.
- Associator is computed on joint velocities (dq/dt), not raw dq, to reduce sampling-jitter sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Any

import math
import numpy as np


# ============================================================
# Data contracts
# ============================================================

@dataclass
class JointSample:
    """
    Normalized joint sample consumed by the engine.

    Attributes
    ----------
    timestamp : float
        Seconds. Prefer monotonic timestamps.
    q : np.ndarray
        Shape (7,), radians.
    source : str
        "csv" or "rws"
    meta : dict
        Optional metadata.
    """
    timestamp: float
    q: np.ndarray
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).reshape(7,)


@dataclass
class AssociatorEvent:
    frame_idx: int
    timestamp: float
    raw_score: float
    normalized_score: float
    tcp_step_mm: Optional[float] = None
    severity: str = "INFO"
    detail: str = ""


@dataclass
class AlarmEvent:
    timestamp: float
    kind: str
    severity: str
    value: float
    threshold: float
    detail: str = ""


@dataclass
class UpdateResult:
    ready: bool
    timestamp: float
    associator_raw: Optional[float] = None
    associator_norm: Optional[float] = None
    tcp_step_mm: Optional[float] = None
    alarms: List[AlarmEvent] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchReport:
    total_samples: int
    associator_scores_raw: np.ndarray
    associator_scores_norm: np.ndarray
    tcp_steps_mm: Optional[np.ndarray]
    associator_events: List[AssociatorEvent]
    alarms: List[AlarmEvent]
    summary: Dict[str, Any]


# ============================================================
# Config
# ============================================================

@dataclass
class YumiThresholds:
    associator_alert_z: float = 3.0
    associator_warn_z: float = 2.0
    tcp_step_mm_alert: float = 5.0
    joint_jump_rad_alert: float = 0.20
    time_gap_alert_s: float = 0.30


@dataclass
class YumiEngineConfig:
    thresholds: YumiThresholds = field(default_factory=YumiThresholds)

    # live mode rolling window for latest samples
    window_size: int = 8

    # associator needs 4 samples -> 3 segment velocities
    min_samples_for_associator: int = 4

    # if true, compute nominal FK and estimated TCP step
    enable_nominal_fk: bool = True

    # robust normalization settings
    zscore_eps: float = 1e-8

    # dt normalization
    min_dt_s: float = 1e-4

    # history caps to avoid unbounded growth in live mode
    associator_history_size: int = 512
    tcp_history_size: int = 512

    # weighting for order-sensitive nonlinear composition
    redundancy_joint_index: int = 6      # q7
    redundancy_gain: float = 1.5
    sign_flip_gain: float = 0.5
    neighbor_coupling_gain: float = 0.15

    # robust baseline statistics
    baseline_window: int = 50


# ============================================================
# YuMi nominal kinematics placeholder
# ============================================================

class YumiNominalKinematics:
    """
    Minimal placeholder for YuMi nominal FK.

    Replace with a proper nominal FK in kinematics/yumi.py later.
    This placeholder keeps the engine API stable, but its TCP output must be
    treated as estimated and non-authoritative.
    """

    def __init__(self) -> None:
        self.name = "ABB YuMi nominal FK (placeholder)"

    def fk_tcp(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(7,)

        # Placeholder pseudo-FK.
        x = 0.15 * math.cos(q[0]) + 0.12 * math.cos(q[0] + q[1]) + 0.08 * math.cos(q[0] + q[1] + q[2])
        y = 0.15 * math.sin(q[0]) + 0.12 * math.sin(q[0] + q[1]) + 0.08 * math.sin(q[0] + q[1] + q[2])
        z = 0.10 * math.sin(q[1]) + 0.08 * math.sin(q[1] + q[2]) + 0.06 * math.sin(q[1] + q[2] + q[3])

        return np.array([x, y, z], dtype=float)


# ============================================================
# Engine
# ============================================================

class SIPAYuMiEngine:
    """
    Unified SIPA audit core for ABB YuMi.

    Live mode:
        engine.update(sample)

    Offline mode:
        engine.analyze_batch(samples)
    """

    def __init__(
        self,
        config: Optional[YumiEngineConfig] = None,
        kinematics: Optional[YumiNominalKinematics] = None
    ) -> None:
        self.cfg = config or YumiEngineConfig()
        self.kin = kinematics or YumiNominalKinematics()

        self._samples: Deque[JointSample] = deque(maxlen=self.cfg.window_size)
        self._assoc_history: Deque[float] = deque(maxlen=self.cfg.associator_history_size)
        self._tcp_history: Deque[np.ndarray] = deque(maxlen=self.cfg.tcp_history_size)
        self._frame_counter: int = 0

    # ========================================================
    # Public API
    # ========================================================

    def reset(self) -> None:
        self._samples.clear()
        self._assoc_history.clear()
        self._tcp_history.clear()
        self._frame_counter = 0

    def update(self, sample: JointSample) -> UpdateResult:
        self._validate_sample(sample)
        self._samples.append(sample)
        self._frame_counter += 1

        alarms: List[AlarmEvent] = []

        self._check_time_gap(alarms)
        self._check_joint_jump(alarms)

        tcp_step_mm = None
        if self.cfg.enable_nominal_fk:
            tcp_step_mm = self._update_tcp_step(sample, alarms)

        if len(self._samples) < self.cfg.min_samples_for_associator:
            return UpdateResult(
                ready=False,
                timestamp=sample.timestamp,
                tcp_step_mm=tcp_step_mm,
                alarms=alarms,
                debug={"reason": "warming_up"}
            )

        raw_score = self._compute_latest_associator_raw()
        self._assoc_history.append(raw_score)
        norm_score = self._normalize_latest_score(raw_score)

        self._check_associator_alarm(sample, raw_score, norm_score, alarms)

        return UpdateResult(
            ready=True,
            timestamp=sample.timestamp,
            associator_raw=raw_score,
            associator_norm=norm_score,
            tcp_step_mm=tcp_step_mm,
            alarms=alarms,
            debug={
                "window_size": len(self._samples),
                "frame_idx": self._frame_counter - 1,
            }
        )

    def analyze_batch(self, samples: List[JointSample]) -> BatchReport:
        self.reset()

        assoc_raw: List[float] = []
        assoc_norm: List[float] = []
        tcp_steps_mm: List[float] = []
        alarms: List[AlarmEvent] = []
        assoc_events: List[AssociatorEvent] = []

        for idx, sample in enumerate(samples):
            result = self.update(sample)

            tcp_steps_mm.append(result.tcp_step_mm if result.tcp_step_mm is not None else np.nan)
            alarms.extend(result.alarms)

            if result.ready:
                assoc_raw.append(result.associator_raw if result.associator_raw is not None else np.nan)
                assoc_norm.append(result.associator_norm if result.associator_norm is not None else np.nan)

                if result.associator_norm is not None and result.associator_norm >= self.cfg.thresholds.associator_warn_z:
                    assoc_events.append(
                        AssociatorEvent(
                            frame_idx=idx,
                            timestamp=sample.timestamp,
                            raw_score=result.associator_raw or 0.0,
                            normalized_score=result.associator_norm,
                            tcp_step_mm=result.tcp_step_mm,
                            severity="ALERT" if result.associator_norm >= self.cfg.thresholds.associator_alert_z else "WARN",
                            detail="Joint-space NARH / associator peak"
                        )
                    )

        assoc_raw_arr = np.asarray(assoc_raw, dtype=float)
        assoc_norm_arr = np.asarray(assoc_norm, dtype=float)
        tcp_steps_arr = np.asarray(tcp_steps_mm, dtype=float)

        summary = self._build_summary(
            total_samples=len(samples),
            assoc_raw=assoc_raw_arr,
            assoc_norm=assoc_norm_arr,
            tcp_steps_mm=tcp_steps_arr,
            alarms=alarms,
            assoc_events=assoc_events,
        )

        return BatchReport(
            total_samples=len(samples),
            associator_scores_raw=assoc_raw_arr,
            associator_scores_norm=assoc_norm_arr,
            tcp_steps_mm=tcp_steps_arr,
            associator_events=assoc_events,
            alarms=alarms,
            summary=summary,
        )

    # ========================================================
    # Internal helpers
    # ========================================================

    def _validate_sample(self, sample: JointSample) -> None:
        if sample.q.shape != (7,):
            raise ValueError(f"Expected q shape (7,), got {sample.q.shape}")
        if not np.all(np.isfinite(sample.q)):
            raise ValueError("Joint sample contains non-finite values")
        if not np.isfinite(sample.timestamp):
            raise ValueError("Invalid timestamp")

    def _check_time_gap(self, alarms: List[AlarmEvent]) -> None:
        if len(self._samples) < 2:
            return

        prev = self._samples[-2]
        curr = self._samples[-1]
        dt = curr.timestamp - prev.timestamp

        if dt <= 0:
            alarms.append(
                AlarmEvent(
                    timestamp=curr.timestamp,
                    kind="time_monotonicity",
                    severity="WARN",
                    value=dt,
                    threshold=0.0,
                    detail="Non-increasing timestamp detected"
                )
            )

        if dt > self.cfg.thresholds.time_gap_alert_s:
            alarms.append(
                AlarmEvent(
                    timestamp=curr.timestamp,
                    kind="time_gap",
                    severity="WARN",
                    value=dt,
                    threshold=self.cfg.thresholds.time_gap_alert_s,
                    detail="Large sample gap may reduce diagnosis confidence"
                )
            )

    def _check_joint_jump(self, alarms: List[AlarmEvent]) -> None:
        if len(self._samples) < 2:
            return

        prev = self._samples[-2]
        curr = self._samples[-1]
        dq = curr.q - prev.q
        peak_jump = float(np.max(np.abs(dq)))

        if peak_jump > self.cfg.thresholds.joint_jump_rad_alert:
            alarms.append(
                AlarmEvent(
                    timestamp=curr.timestamp,
                    kind="joint_jump",
                    severity="WARN",
                    value=peak_jump,
                    threshold=self.cfg.thresholds.joint_jump_rad_alert,
                    detail="Large single-step joint change detected"
                )
            )

    def _update_tcp_step(self, sample: JointSample, alarms: List[AlarmEvent]) -> Optional[float]:
        tcp = self.kin.fk_tcp(sample.q)
        self._tcp_history.append(tcp)

        if len(self._tcp_history) < 2:
            return None

        d = np.linalg.norm(self._tcp_history[-1] - self._tcp_history[-2]) * 1000.0

        if d > self.cfg.thresholds.tcp_step_mm_alert:
            alarms.append(
                AlarmEvent(
                    timestamp=sample.timestamp,
                    kind="tcp_step",
                    severity="WARN",
                    value=d,
                    threshold=self.cfg.thresholds.tcp_step_mm_alert,
                    detail="Estimated TCP step exceeds nominal YuMi threshold"
                )
            )

        return d

    def _compute_latest_associator_raw(self) -> float:
        """
        Compute latest raw associator score from the last 4 samples.

        Uses three segment velocities (dq/dt), not raw dq, to reduce false peaks
        caused purely by uneven sampling intervals.
        """
        s0, s1, s2, s3 = list(self._samples)[-4:]

        v1 = self._segment_velocity(s0, s1)
        v2 = self._segment_velocity(s1, s2)
        v3 = self._segment_velocity(s2, s3)

        left = self._compose(self._compose(v1, v2), v3)
        right = self._compose(v1, self._compose(v2, v3))

        associator = left - right
        raw_score = float(np.linalg.norm(associator))
        return raw_score

    def _segment_velocity(self, a: JointSample, b: JointSample) -> np.ndarray:
        dt = max(float(b.timestamp - a.timestamp), self.cfg.min_dt_s)
        return (b.q - a.q) / dt

    def _compose(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Order-sensitive nonlinear composition over segment velocities.

        Purpose
        -------
        Make hidden branch switching / order instability visible in 7DoF joint streams.

        This is not ordinary vector addition.
        """
        a = np.asarray(a, dtype=float).reshape(7,)
        b = np.asarray(b, dtype=float).reshape(7,)

        out = a + b

        for i in range(6):
            coupling = self.cfg.neighbor_coupling_gain * (a[i] * b[i + 1] - b[i] * a[i + 1])
            out[i] += coupling
            out[i + 1] -= coupling

        sign_flip = np.sign(a) * np.sign(b) < 0
        out += self.cfg.sign_flip_gain * sign_flip.astype(float) * np.minimum(np.abs(a), np.abs(b))

        r = self.cfg.redundancy_joint_index
        out[r] += self.cfg.redundancy_gain * np.dot(a[:6], b[:6]) * np.sign(a[r] + b[r] + 1e-12)

        return out

    def _normalize_latest_score(self, raw_score: float) -> float:
        hist = list(self._assoc_history)[-self.cfg.baseline_window:]
        if len(hist) < 5:
            return raw_score

        arr = np.asarray(hist, dtype=float)
        mu = np.median(arr)
        sigma = np.median(np.abs(arr - mu)) * 1.4826 + self.cfg.zscore_eps

        return float((raw_score - mu) / sigma)

    def _check_associator_alarm(
        self,
        sample: JointSample,
        raw_score: float,
        norm_score: float,
        alarms: List[AlarmEvent]
    ) -> None:
        if norm_score >= self.cfg.thresholds.associator_alert_z:
            alarms.append(
                AlarmEvent(
                    timestamp=sample.timestamp,
                    kind="associator_peak",
                    severity="ALERT",
                    value=norm_score,
                    threshold=self.cfg.thresholds.associator_alert_z,
                    detail="Joint-space NARH peak detected; possible hidden branch switch / ordering instability"
                )
            )
        elif norm_score >= self.cfg.thresholds.associator_warn_z:
            alarms.append(
                AlarmEvent(
                    timestamp=sample.timestamp,
                    kind="associator_peak",
                    severity="WARN",
                    value=norm_score,
                    threshold=self.cfg.thresholds.associator_warn_z,
                    detail="Elevated joint-space NARH score"
                )
            )

    def _build_summary(
        self,
        total_samples: int,
        assoc_raw: np.ndarray,
        assoc_norm: np.ndarray,
        tcp_steps_mm: np.ndarray,
        alarms: List[AlarmEvent],
        assoc_events: List[AssociatorEvent],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_samples": total_samples,
            "associator_event_count": len(assoc_events),
            "alarm_count": len(alarms),
        }

        if assoc_raw.size > 0 and np.any(np.isfinite(assoc_raw)):
            peak_idx = int(np.nanargmax(assoc_norm)) if assoc_norm.size > 0 and np.any(np.isfinite(assoc_norm)) else -1
            summary.update({
                "associator_raw_peak": float(np.nanmax(assoc_raw)),
                "associator_norm_peak": float(np.nanmax(assoc_norm)) if assoc_norm.size > 0 and np.any(np.isfinite(assoc_norm)) else float("nan"),
                "associator_peak_relative_index": peak_idx,
            })

        if tcp_steps_mm.size > 0 and np.any(np.isfinite(tcp_steps_mm)):
            summary["tcp_step_mm_peak"] = float(np.nanmax(tcp_steps_mm))

        return summary


# ============================================================
# Convenience helpers for adapters
# ============================================================

def detect_unit_from_joint_matrix(q_matrix: np.ndarray) -> str:
    q_matrix = np.asarray(q_matrix, dtype=float)
    max_abs = float(np.nanmax(np.abs(q_matrix)))
    return "deg" if max_abs > 6.5 else "rad"


def convert_joint_matrix_to_rad(q_matrix: np.ndarray, unit: str) -> np.ndarray:
    q_matrix = np.asarray(q_matrix, dtype=float)

    if unit == "deg":
        return np.deg2rad(q_matrix)
    if unit == "rad":
        return q_matrix

    raise ValueError(f"Unsupported unit: {unit}")


def build_joint_samples(
    q_matrix: np.ndarray,
    timestamps: np.ndarray,
    source: str,
    meta_list: Optional[List[Dict[str, Any]]] = None,
) -> List[JointSample]:
    q_matrix = np.asarray(q_matrix, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if q_matrix.ndim != 2 or q_matrix.shape[1] != 7:
        raise ValueError(f"Expected q_matrix shape (N, 7), got {q_matrix.shape}")
    if len(q_matrix) != len(timestamps):
        raise ValueError("q_matrix and timestamps must have same length")

    if meta_list is None:
        meta_list = [{} for _ in range(len(q_matrix))]

    out: List[JointSample] = []
    for i in range(len(q_matrix)):
        out.append(
            JointSample(
                timestamp=float(timestamps[i]),
                q=q_matrix[i],
                source=source,
                meta=meta_list[i],
            )
        )
    return out


# ============================================================
# Optional self-test
# ============================================================

if __name__ == "__main__":
    engine = SIPAYuMiEngine()

    ts = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.90])
    q = np.array([
        [0.00, -0.30, 0.20, 0.10, 0.00, 0.20, 0.10],
        [0.01, -0.29, 0.21, 0.10, 0.00, 0.20, 0.11],
        [0.02, -0.28, 0.22, 0.11, 0.01, 0.21, 0.12],
        [0.03, -0.27, 0.23, 0.12, 0.02, 0.22, 0.60],  # q7 sudden change
        [0.04, -0.26, 0.24, 0.13, 0.03, 0.23, 0.62],
        [0.05, -0.25, 0.25, 0.14, 0.04, 0.24, 0.64],  # larger dt to test normalization
    ])

    samples = build_joint_samples(q, ts, source="csv")
    report = engine.analyze_batch(samples)

    print("=" * 60)
    print("SIPA YuMi Engine Smoke Test")
    print("=" * 60)
    print(report.summary)

    for ev in report.associator_events[:5]:
        print(
            f"[{ev.severity}] frame={ev.frame_idx} "
            f"t={ev.timestamp:.3f}s "
            f"score={ev.normalized_score:.3f} "
            f"tcp_step={ev.tcp_step_mm}"
        )
