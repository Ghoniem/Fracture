"""Shared utilities used by every validation_*.py module in Uvalidation."""
from __future__ import annotations


def err_pct(num: float, ana: float, eps: float = 1e-14) -> float:
    """Relative percentage error, NaN when the analytical value cannot anchor one.

    Returns ``100 * (num - ana) / ana`` when ``|ana| > eps``, otherwise NaN.
    Returning NaN (rather than a fabricated tiny denominator) keeps cases like
    pure-mode-I KII validation honest: downstream consumers already use
    nanmean/nanstd and will simply skip these rows.
    """
    if abs(ana) <= eps:
        return float("nan")
    return float(100.0 * (num - ana) / ana)
