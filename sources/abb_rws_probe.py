#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sources/abb_rws_probe.py

Minimal ABB RWS alarm-only probe for SIPA YuMi.
Feeds JointSample objects into core.sipa_yumi_engine.SIPAYuMiEngine.

MVP design
----------
- poll-based (simpler than websocket subscription)
- alarm-only
- configurable joint target endpoint
- supports JSON first, XML fallback parser stub
- supports local debug JSON replay:
    * single payload object
    * sequence payload list (frame-by-frame replay)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from requests.auth import HTTPDigestAuth

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.sipa_yumi_engine import JointSample, SIPAYuMiEngine

JsonLike = Union[Dict[str, Any], List[Dict[str, Any]]]


class ABBRWSProbe:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        joint_path: str,
        poll_interval_s: float = 0.20,
        verify_tls: bool = False,
        timeout_s: float = 3.0,
        raw_unit: str = "deg",
        debug_json: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.joint_path = joint_path
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.raw_unit = raw_unit

        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(username, password)
        self.session.verify = verify_tls
        self.session.headers.update({"Accept": "application/json"})

        self._t0_wall = time.monotonic()

        # debug replay state
        self.frame_index = 0
        self.mock_data: Optional[JsonLike] = None
        self.debug_mode = False
        if debug_json:
            self.debug_mode = True
            debug_path = Path(debug_json)
            self.mock_data = json.loads(debug_path.read_text(encoding="utf-8"))

    def _url(self) -> str:
        if self.joint_path.startswith("http://") or self.joint_path.startswith("https://"):
            return self.joint_path
        return f"{self.base_url}{self.joint_path}"

    def _fetch_raw(self) -> str:
        """
        Fetch raw payload text.

        In debug mode:
        - dict payload  -> replay same frame every cycle
        - list payload  -> replay one frame per cycle, wrapping around
        """
        if self.mock_data is not None:
            if isinstance(self.mock_data, list):
                if len(self.mock_data) == 0:
                    raise ValueError("debug-json contains an empty list")
                data = self.mock_data[self.frame_index]
                self.frame_index = (self.frame_index + 1) % len(self.mock_data)
                return json.dumps(data)
            return json.dumps(self.mock_data)

        resp = self.session.get(self._url(), timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.text

    def fetch_joint_sample(self) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        raw_text = self._fetch_raw()

        timestamp = time.monotonic() - self._t0_wall
        meta: Dict[str, Any] = {
            "endpoint": self.joint_path,
            "debug_source": "json_replay" if self.mock_data is not None else None,
        }

        stripped = raw_text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            payload = json.loads(raw_text)
            if isinstance(payload, list):
                raise ValueError("Expected a single JSON object after _fetch_raw replay selection, got list")
            q = self._extract_joints_from_json(payload)
        else:
            q = self._extract_joints_from_xml(raw_text)

        if self.raw_unit == "deg":
            q = np.deg2rad(q)
            meta["raw_unit"] = "deg"
        else:
            meta["raw_unit"] = "rad"

        return timestamp, q, meta

    def _extract_joints_from_json(self, payload: Dict[str, Any]) -> np.ndarray:
        for key in ("joints", "jointtarget", "rob_joints", "robtargetjoints"):
            if key in payload and isinstance(payload[key], (list, tuple)) and len(payload[key]) >= 7:
                return np.asarray(payload[key][:7], dtype=float)

        state = payload.get("state", payload)
        candidates = []

        for name in ("rax_1", "rax_2", "rax_3", "rax_4", "rax_5", "rax_6", "eax_a"):
            if isinstance(state, dict) and name in state:
                candidates.append(float(state[name]))
        if len(candidates) == 7:
            return np.asarray(candidates, dtype=float)

        embedded = payload.get("_embedded", {})
        for _, value in embedded.items():
            if isinstance(value, list):
                flat = {}
                for item in value:
                    if isinstance(item, dict):
                        flat.update(item)
                candidates = []
                for name in ("rax_1", "rax_2", "rax_3", "rax_4", "rax_5", "rax_6", "eax_a"):
                    if name in flat:
                        candidates.append(float(flat[name]))
                if len(candidates) == 7:
                    return np.asarray(candidates, dtype=float)

        raise ValueError(
            "Could not extract 7 joint values from RWS JSON payload. "
            "Save one payload sample and adapt _extract_joints_from_json()."
        )

    def _extract_joints_from_xml(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "XML parsing not implemented in MVP. "
            "Use a JSON-capable RWS endpoint first, or extend this function."
        )

    def run(self, engine: SIPAYuMiEngine, duration_s: Optional[float] = None, prefix: str = "RWS") -> None:
        start = time.monotonic()

        while True:
            ts, q, meta = self.fetch_joint_sample()
            sample = JointSample(timestamp=ts, q=q, source="rws_debug" if self.debug_mode else "rws", meta=meta)
            result = engine.update(sample)

            if result.ready:
                alarms = ", ".join(f"{a.severity}:{a.kind}={a.value:.3f}" for a in result.alarms) or "none"
                assoc = result.associator_norm if result.associator_norm is not None else float("nan")
                tcp = result.tcp_step_mm if result.tcp_step_mm is not None else float("nan")
                print(
                    f"[{prefix}] t={result.timestamp:8.3f}s "
                    f"assoc={assoc:8.3f} "
                    f"tcp_step_mm={tcp:8.3f} "
                    f"alarms={alarms}"
                )
            else:
                print(f"[{prefix}] t={result.timestamp:8.3f}s warming_up")

            if duration_s is not None and (time.monotonic() - start) >= duration_s:
                break

            time.sleep(self.poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None, help="Controller base URL, e.g. https://127.0.0.1")
    parser.add_argument("--user", default="Default User")
    parser.add_argument("--password", default="robotics")
    parser.add_argument(
        "--joint-path",
        default=None,
        help="RWS joint target resource path, e.g. /rw/.../jointtarget?json=1"
    )
    parser.add_argument("--debug-json", default=None, help="Read a saved RWS JSON payload or sequence from local file instead of calling the network")
    parser.add_argument("--poll", type=float, default=0.20, help="Polling interval in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds")
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--unit", choices=["deg", "rad"], default="deg")

    args = parser.parse_args()

    engine = SIPAYuMiEngine()

    if args.debug_json:
        # infer endpoint if payload is a dict with _links.self.href, else use fallback
        joint_path = "/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1"
        try:
            debug_obj = json.loads(Path(args.debug_json).read_text(encoding="utf-8"))
            if isinstance(debug_obj, dict):
                joint_path = debug_obj.get("_links", {}).get("self", {}).get("href", joint_path)
        except Exception:
            pass

        probe = ABBRWSProbe(
            base_url=args.host or "https://debug.local",
            username=args.user,
            password=args.password,
            joint_path=args.joint_path or joint_path,
            poll_interval_s=args.poll,
            verify_tls=args.verify_tls,
            raw_unit=args.unit,
            debug_json=args.debug_json,
        )
        probe.run(engine, duration_s=args.duration, prefix="RWS-DEBUG")
        return

    if not args.host or not args.joint_path:
        parser.error("Either provide --debug-json, or provide both --host and --joint-path.")

    probe = ABBRWSProbe(
        base_url=args.host,
        username=args.user,
        password=args.password,
        joint_path=args.joint_path,
        poll_interval_s=args.poll,
        verify_tls=args.verify_tls,
        raw_unit=args.unit,
    )
    probe.run(engine, duration_s=args.duration, prefix="RWS")


if __name__ == "__main__":
    main()
