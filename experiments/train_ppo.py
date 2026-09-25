from __future__ import annotations

import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.config import load_config
from src.envs.traffic_env import TrafficEnv


LAMBDA_SCALES = [0.75, 1.0, 1.25]


def make_env(cfg, seed: int, lam_scale: float):
    def _init():
        local_cfg = deepcopy(cfg)
        for a in local_cfg.traffic.arrival_rate:
            local_cfg.traffic.arrival_rate[a] *= lam_scale
        env = TrafficEnv(cfg=local_cfg, seed=seed)
        env = Monitor(env, filename=None)
        return env
    return _init


def main() -> None:
    print(">>> ВОШЛИ В main()")
    cfg = load_config()
    print(">>> config загружен, total_timesteps =", cfg.rl.total_timesteps)
    print(">>> c_switch =", cfg.signal.c_switch)
    print(">>> domain randomization scales =", LAMBDA_SCALES)

    Path(cfg.paths.models).mkdir(parents=True, exist_ok=True)
    Path(cfg.rl.tensorboard_log).mkdir(parents=True, exist_ok=True)

    n_envs = 6
    scales = [random.Random(seed).choice(LAMBDA_SCALES) for seed in range(n_envs)]
    print(">>> масштабы λ по средам:", scales)

    venv = DummyVecEnv([
        make_env(cfg, seed=i, lam_scale=scales[i]) for i in range(n_envs)
    ])

    print(">>> создаём новый VecNormalize")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print(">>> Создаём новую модель PPO с нуля")
    model = PPO(
        policy=cfg.rl.policy,
        env=venv,
        learning_rate=cfg.rl.learning_rate,
        n_steps=cfg.rl.n_steps,
        batch_size=cfg.rl.batch_size,
        n_epochs=cfg.rl.n_epochs,
        gamma=cfg.rl.gamma,
        gae_lambda=cfg.rl.gae_lambda,
        clip_range=cfg.rl.clip_range,
        ent_coef=cfg.rl.ent_coef,
        verbose=cfg.rl.verbose,
        tensorboard_log=cfg.rl.tensorboard_log,
        seed=0,
        device="cpu",
    )

    print(f">>> Начинаем обучение PPO на {cfg.rl.total_timesteps} шагов...")
    t0 = time.time()
    model.learn(
        total_timesteps=cfg.rl.total_timesteps,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    dt = time.time() - t0
    print(f">>> Обучение завершено за {dt:.1f} с ({dt / 60:.1f} мин)")

    model.save(cfg.rl.model_path)
    venv.save(str(Path(cfg.paths.models) / "vec_normalize.pkl"))
    print(f">>> Модель сохранена: {cfg.rl.model_path}")

    venv.close()
    print(">>> main() завершена")


if __name__ == "__main__":
    print(">>> точка входа достигнута")
    main()