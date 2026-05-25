"""Smoothing helpers used by deformed crack plots."""
from __future__ import annotations
import numpy as np

def moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, float).reshape(-1,)
    window = int(window)
    if window <= 1 or y.size == 0:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2

    yp = np.pad(y, pad_width=pad, mode="reflect")
    valid = np.isfinite(yp).astype(float)
    yp0 = np.where(np.isfinite(yp), yp, 0.0)

    k = np.ones(window, float)
    num = np.convolve(yp0, k, mode="valid")
    den = np.convolve(valid, k, mode="valid")
    den = np.where(den > 0, den, np.nan)
    return num / den


def resolve_smooth_window(value, n_crack_elements: int, *, C: int = 50, wmin: int = 3) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            nh = max(1, int(n_crack_elements))
            w = int(round(float(C) / float(nh)))
            w = max(int(wmin), w)
        else:
            try:
                w = int(round(float(value)))
            except Exception:
                return 0
    else:
        try:
            w = int(round(float(value)))
        except Exception:
            return 0

    if w <= 1:
        return 0
    if w % 2 == 0:
        w += 1
    return int(w)
