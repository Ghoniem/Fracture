"""Result containers and diagnostics for crack propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

@dataclass
class TrialRecord:
    f: float
    delta_a: float
    theta: float
    keff: float
    accepted: bool
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TipPropagationReport:
    pid: int
    which: str
    grew: bool
    reason: str
    Kc: float
    KI: float
    KII: float
    keff: float
    theta0: float
    f_accepted: Optional[float] = None
    delta_a: Optional[float] = None
    theta_accepted: Optional[float] = None
    trials: List[TrialRecord] = field(default_factory=list)

@dataclass
class PropagationResult:
    network_new: Any
    base_solution: Dict[str, Any]
    reports: List[TipPropagationReport] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
