"""Result containers for crack propagation.

Only ``TrialRecord`` is used by the propagation pipeline today (consumed by
``step_control``). The propagator emits its own ``TipPropagationReport`` and
``PropagationResult`` dataclasses; the unused duplicates that previously lived
here were deleted to remove the confusion of two same-named dataclasses with
incompatible field signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TrialRecord:
    f: float
    delta_a: float
    theta: float
    keff: float
    accepted: bool
    meta: Dict[str, Any] = field(default_factory=dict)
