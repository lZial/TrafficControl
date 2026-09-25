from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.envs.traffic_env import TrafficEnv


def main():
    cfg = load_config()
    env = TrafficEnv(cfg=cfg, seed=0)

    obs, info = env.reset(seed=0)
    print(f"После reset: obs.shape={obs.shape}, info={info}")
    print(f"  sim.env.now={env.sim.env.now}, arrived={env.sim.total_arrived}, served={env.sim.total_served}")
    print(f"  активных процессов в SimPy: {len(env.sim.env._queue)}")

    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        if i % 10 == 0:
            print(
                f"step {i+1}: sim.env.now={env.sim.env.now:.0f}, "
                f"arrived={env.sim.total_arrived}, served={env.sim.total_served}, "
                f"q_total={info['q_total']:.0f}, phase={info['phase']}"
            )

    print(f"\nПосле 50 шагов: arrived={env.sim.total_arrived}, served={env.sim.total_served}")
    print(f"total_reward={total_reward:.2f}")


if __name__ == "__main__":
    main()