"""Fracture toughness fields for propagation decisions.

Upropagation uses a minimal interface: Kc(x,y) -> float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

class ToughnessField(Protocol):
    def Kc(self, x: float, y: float) -> float:
        """Return fracture toughness Kc at position (x,y)."""

@dataclass(frozen=True)
class ConstantToughness:
    Kc0: float
    def Kc(self, x: float, y: float) -> float:
        _ = (x, y)
        return float(self.Kc0)

@dataclass(frozen=True)
class CallableToughness:
    func: Callable[[float, float], float]
    def Kc(self, x: float, y: float) -> float:
        return float(self.func(float(x), float(y)))
