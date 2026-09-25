from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from src.config import load_config
from src.envs.traffic_env import TrafficEnv


def normalize_obs(obs, mean, var, clip=10.0):
    norm = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(norm, -clip, clip).astype(np.float32)


def run_episode(env, model, obs_mean, obs_var, seed):
    obs, info = env.reset(seed=seed)
    obs_norm = normalize_obs(obs, obs_mean, obs_var)

    actions_count = {0: 0, 1: 0}
    switches = 0
    q_history = []

    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs_norm, deterministic=True)
        action = int(action)
        actions_count[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        obs_norm = normalize_obs(obs, obs_mean, obs_var)

        if info["switched"]:
            switches += 1
        q_history.append(info["q_total"])

        done = terminated or truncated
        steps += 1

    return steps, actions_count, switches, q_history


def main():
    cfg = load_config()
    model = PPO.load(cfg.rl.model_path, device="cpu")

    mp = Path(cfg.paths.models)
    obs_mean = np.load(mp / "obs_mean.npy")
    obs_var = np.load(mp / "obs_var.npy")

    print("Диагностика обученного PPO")
    print("=" * 60)

    for seed in [0, 1, 2]:
        env = TrafficEnv(cfg=cfg, seed=seed)
        steps, actions, switches, q_hist = run_episode(env, model, obs_mean, obs_var, seed)
        summary = env.episode_summary()
        env.close()

        a0_frac = actions[0] / steps
        a1_frac = actions[1] / steps
        avg_q = np.mean(q_hist)
        max_q = np.max(q_hist)

        print(f"\n--- seed={seed} ---")
        print(f"  steps           : {steps}")
        print(f"  actions         : keep={actions[0]} ({a0_frac:.1%}), switch={actions[1]} ({a1_frac:.1%})")
        print(f"  switches        : {switches}  (раз за эпизод)")
        print(f"  avg q_total     : {avg_q:.2f}")
        print(f"  max q_total     : {max_q}")
        print(f"  W               : {summary['W']:.2f} с")
        print(f"  Lmax            : {summary['Lmax']}")
        print(f"  throughput      : {summary['throughput']:.3f}")
        print(f"  served/arrived  : {summary['served_total']}/{summary['arrived_total']}")


if __name__ == "__main__":
    main()