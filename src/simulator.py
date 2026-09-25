from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import simpy

from .config import Config
from .intersection import Intersection

Controller = Callable[[Intersection, simpy.Environment], int]


class TrafficSimulator:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.inter = Intersection(cfg=cfg)

        self.history: List[Dict] = []

        self.total_arrived = 0
        self.total_served = 0

        self.total_delay = 0.0

        self.total_wait_per_vehicle = 0.0

        self.max_total_queue = 0

        self._serve_credit = 0.0

    def _arrival_process(self, approach: str, rate: float):
        while True:
            dt = self.rng.exponential(1.0 / rate)
            yield self.env.timeout(dt)
            self.inter.add_arrival(approach, 1)
            self.total_arrived += 1

    def _start_processes(self):
        for a in self.cfg.topology.approaches:
            rate = self.cfg.traffic.arrival_rate[a]
            if rate > 0:
                self.env.process(self._arrival_process(a, rate))

    def step(self, action: int) -> Dict:
        switched = False
        if action == 1:
            switched = self.inter.switch_phase()

        self._serve_credit += self.cfg.traffic.service_rate * self.cfg.simulation.dt
        n_to_serve = int(self._serve_credit)     
        self._serve_credit -= n_to_serve      

        served_now = self.inter.serve(n_to_serve)
        self.total_served += served_now

        q_total = self.inter.total_queue()
        self.total_delay += q_total * self.cfg.simulation.dt
        self.total_wait_per_vehicle += q_total * self.cfg.simulation.dt
        if q_total > self.max_total_queue:
            self.max_total_queue = q_total

        self.inter.tick(self.cfg.simulation.dt)

        self.env.run(until=self.env.now + self.cfg.simulation.dt)

        snap = {
            "t": self.env.now,
            "phase": self.inter.phase,
            "t_phase": self.inter.t_phase,
            "queues": dict(self.inter.queues),
            "q_total": q_total,
            "switched": switched,
            "arrived_total": self.total_arrived,
            "served_total": self.total_served,
            "served_now": served_now,
        }
        self.history.append(snap)
        return snap

    def current_observation(self) -> list:
        return self.inter.to_state_vector()

    def run(self, controller: Controller, horizon: Optional[int] = None) -> Dict:
        H = horizon if horizon is not None else self.cfg.simulation.horizon

        self._start_processes()

        for _ in range(H):
            action = controller(self.inter, self.env)
            self.step(action)

        return self.summary()

    def summary(self) -> Dict:
        W = (self.total_wait_per_vehicle / self.total_served) if self.total_served > 0 else 0.0
        horizon_seconds = self.cfg.simulation.dt * len(self.history) if self.history else 0.0
        throughput = (self.total_served / horizon_seconds) if horizon_seconds > 0 else 0.0

        return {
            "W": W,
            "Lmax": self.max_total_queue,
            "throughput": throughput,
            "total_delay": self.total_delay,
            "arrived_total": self.total_arrived,
            "served_total": self.total_served,
            "n_steps": len(self.history),
            "seed": self.seed,
        }