from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config, load_config
from src.simulator import TrafficSimulator


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[Config] = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self._seed = seed

        n_features = 4 + 4 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

        self.sim: Optional[TrafficSimulator] = None
        self._step_count = 0
        self._max_steps = self.cfg.simulation.horizon
        self._c_switch = self.cfg.signal.c_switch
        self._dt = self.cfg.simulation.dt

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self.sim = TrafficSimulator(self.cfg, seed=self._seed)
        self.sim._start_processes()
        self._step_count = 0

        obs = np.asarray(self.sim.current_observation(), dtype=np.float32)
        info: Dict[str, Any] = {"t": self.sim.env.now, "phase": self.sim.inter.phase}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Сначала вызови reset().")

        action = int(action)

        q_before = self.sim.inter.total_queue()
        snap = self.sim.step(action)
        switched = bool(snap["switched"])

        q_total = float(q_before)
        reward = -(q_total * self._dt) - (self._c_switch if switched else 0.0)

        q_after = float(self.sim.inter.total_queue())
        reward += 0.5 * (q_total - q_after)

        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        truncated = False

        obs = np.asarray(self.sim.current_observation(), dtype=np.float32)
        info: Dict[str, Any] = {
            "t": self.sim.env.now,
            "phase": self.sim.inter.phase,
            "t_phase": self.sim.inter.t_phase,
            "q_total": q_total,
            "switched": switched,
            "served_total": self.sim.total_served,
            "arrived_total": self.sim.total_arrived,
            "step": self._step_count,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def episode_summary(self) -> Dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Сначала вызови reset().")
        return self.sim.summary()