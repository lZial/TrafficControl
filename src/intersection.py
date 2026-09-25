from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import Config


@dataclass
class Intersection:
    cfg: Config
    queues: Dict[str, int] = field(default_factory=dict)         
    cumulative_wait: Dict[str, float] = field(default_factory=dict)  
    served: Dict[str, int] = field(default_factory=dict)         
    phase: str = ""                                              
    t_phase: float = 0.0                              

    def __post_init__(self):
        for a in self.cfg.topology.approaches:
            self.queues.setdefault(a, 0)
            self.cumulative_wait.setdefault(a, 0.0)
            self.served.setdefault(a, 0)
        if not self.phase:
            self.phase = self.cfg.signal.initial_phase


    def add_arrival(self, approach: str, n: int = 1) -> None:
        self.queues[approach] += n

    def active_green_approaches(self) -> List[str]:
        return self.cfg.topology.phase_green[self.phase]

    def queue_for_phase(self, phase: str) -> int:
        approaches = self.cfg.topology.phase_green[phase]
        return sum(self.queues[a] for a in approaches)

    def queue_for_other_phase(self) -> int:
        phases = self.cfg.topology.phase_names
        other = [p for p in phases if p != self.phase]
        if not other:
            return 0
        return self.queue_for_phase(other[0])

    def total_queue(self) -> int:
        return sum(self.queues.values())

    def serve(self, n_to_serve: int) -> int:
        if n_to_serve <= 0:
            return 0

        green = self.active_green_approaches()
        remaining = n_to_serve
        served_total = 0

        while remaining > 0:
            candidates = [a for a in green if self.queues[a] > 0]
            if not candidates:
                break
            for a in candidates:
                if remaining <= 0:
                    break
                self.queues[a] -= 1
                self.served[a] += 1
                remaining -= 1
                served_total += 1

        return served_total
    def can_switch(self) -> bool:
        return self.t_phase >= self.cfg.signal.min_green

    def switch_phase(self) -> bool:
        if not self.can_switch():
            return False
        phases = self.cfg.topology.phase_names
        idx = phases.index(self.phase)
        self.phase = phases[(idx + 1) % len(phases)]
        self.t_phase = 0.0
        return True

    def tick(self, dt: float) -> None:
        self.t_phase += dt

    def to_state_vector(self) -> List[float]:
        norm = self.cfg.normalization
        phases = self.cfg.topology.phase_names
        state = []
        for a in self.cfg.topology.approaches:
            state.append(self.queues[a] / norm.max_queue)
        for a in self.cfg.topology.approaches:
            state.append(self.cumulative_wait[a] / norm.max_wait)
        state.append(float(phases.index(self.phase)))
        state.append(self.t_phase / self.cfg.signal.max_green)
        state.append(self.t_phase / self.cfg.signal.max_green)
        return state