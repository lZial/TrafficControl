from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.envs.traffic_env import TrafficEnv


def normalize_obs(obs, mean, var, clip=10.0):
    norm = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(norm, -clip, clip).astype(np.float32)


def run_one_episode(cfg, model, obs_mean, obs_var, seed):
    env = TrafficEnv(cfg=cfg, seed=seed)
    obs, info = env.reset(seed=seed)
    obs_norm = normalize_obs(obs, obs_mean, obs_var)

    done = False
    while not done:
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        obs_norm = normalize_obs(obs, obs_mean, obs_var)
        done = terminated or truncated

    summary = env.episode_summary()
    env.close()
    return summary


def main():
    cfg = load_config()

    mp = Path(cfg.paths.models)
    obs_mean = np.load(mp / "obs_mean.npy")
    obs_var = np.load(mp / "obs_var.npy")

    print(">>> Загружаем PPO-модель")
    model = PPO.load(cfg.rl.model_path, device="cpu")

    rows = []
    for seed in tqdm(cfg.simulation.seeds, desc="RL episodes"):
        summary = run_one_episode(cfg, model, obs_mean, obs_var, seed)
        summary["strategy"] = "rl"
        rows.append(summary)

    df = pd.DataFrame(rows)

    metric_cols = ["W", "Lmax", "throughput", "total_delay"]
    print("\n=== RL (PPO): средние за 30 эпизодов ===")
    print(df[metric_cols].agg(["mean", "std"]).round(4))

    cfg_env = os.environ.get("TRAFFIC_CONFIG", "config.yaml")
    tag = Path(cfg_env).stem.replace("config_", "").replace("config", "medium")
    out_dir = Path(cfg.paths.data_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_rl.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()