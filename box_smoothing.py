# box_smoothing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class TrackState:
    """Per-track smoothing state (in center/size space)."""
    cx: float
    cy: float
    w: float
    h: float
    last_frame: int


def xyxy_to_cxcywh(xyxy: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class BoxSmootherEMA:
    """
    Per-track EMA smoothing for detector boxes.

    Key ideas:
      - Smooth in (cx, cy, w, h) space (usually looks better than smoothing xyxy directly).
      - Optionally clamp width/height changes when the object's center is basically stationary,
        which is exactly your "stationary car breathing" symptom.

    Parameters
    ----------
    alpha : float
        EMA factor. Smaller = smoother but more lag (0.10–0.35 typical).
    max_age_frames : int
        Track state is dropped if unseen for this many frames.
    min_wh_px : float
        Ignore boxes smaller than this size.
    stationary_px : float
        If center moves less than this many pixels, treat as stationary for clamping.
    max_size_change_stationary : float
        Max fractional change per frame for w/h when stationary (e.g. 0.02 = ±2%).
        Set to 0 to disable.
    """
    def __init__(
        self,
        alpha: float = 0.20,
        max_age_frames: int = 300,
        min_wh_px: float = 2.0,
        stationary_px: float = 2.0,
        max_size_change_stationary: float = 0.02,
    ):
        self.alpha = float(alpha)
        self.max_age_frames = int(max_age_frames)
        self.min_wh_px = float(min_wh_px)
        self.stationary_px = float(stationary_px)
        self.max_size_change_stationary = float(max_size_change_stationary)
        self.state: Dict[int, TrackState] = {}

    def _valid_xyxy(self, xyxy: np.ndarray) -> bool:
        if xyxy.shape != (4,):
            return False
        if not np.all(np.isfinite(xyxy)):
            return False
        cx, cy, w, h = xyxy_to_cxcywh(xyxy)
        return (w >= self.min_wh_px) and (h >= self.min_wh_px) and np.isfinite(cx) and np.isfinite(cy)

    def _cleanup(self, frame_idx: int) -> None:
        dead = [tid for tid, st in self.state.items() if (frame_idx - st.last_frame) > self.max_age_frames]
        for tid in dead:
            del self.state[tid]

    def update_many(self, frame_idx: int, ids: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        ids: (N,) int
        xyxy: (N,4) float
        returns: (N,4) float32 (smoothed)
        """
        if len(ids) == 0:
            self._cleanup(frame_idx)
            return xyxy.astype(np.float32, copy=True)

        out = xyxy.astype(np.float32, copy=True)

        for i, tid in enumerate(ids):
            tid = int(tid)
            raw = out[i]
            if not self._valid_xyxy(raw):
                continue

            rcx, rcy, rw, rh = xyxy_to_cxcywh(raw)

            st = self.state.get(tid)
            if st is None:
                self.state[tid] = TrackState(rcx, rcy, rw, rh, frame_idx)
                continue

            # Stationary clamp: if the center barely moved, limit w/h changes per frame
            if self.max_size_change_stationary > 0.0:
                dc = np.hypot(rcx - st.cx, rcy - st.cy)
                if dc <= self.stationary_px:
                    frac = self.max_size_change_stationary
                    rw = float(np.clip(rw, st.w * (1.0 - frac), st.w * (1.0 + frac)))
                    rh = float(np.clip(rh, st.h * (1.0 - frac), st.h * (1.0 + frac)))

            # keep center responsive (no lag)
            st.cx = rcx
            st.cy = rcy

            # smooth size
            a = self.alpha
            st.w = a * rw + (1.0 - a) * st.w
            st.h = a * rh + (1.0 - a) * st.h
            st.last_frame = frame_idx

            out[i] = cxcywh_to_xyxy(st.cx, st.cy, st.w, st.h)


        self._cleanup(frame_idx)
        return out
