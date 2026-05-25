"""Shared context for propagation runs.

Optional place for caching, solver-call counters, and logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class PropagationContext:
    counters: Dict[str, int] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0)) + int(n)
