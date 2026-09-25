import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.env_checker import check_env
from src.envs.traffic_env import TrafficEnv
from src.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    env = TrafficEnv(cfg=cfg, seed=0)
    print("Проверяем среду через SB3 env_checker...")
    check_env(env, warn=True, skip_render_check=True)
    print("OK: среда прошла проверку")

    obs, info = env.reset(seed=0)
    print(f"reset: obs.shape={obs.shape}, dtype={obs.dtype}, info={info}")

    total_r = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        print(
            f"step {i + 1}: action={action}, r={r:.2f}, "
            f"q_total={info['q_total']:.0f}, phase={info['phase']}, switched={info['switched']}"
        )
    print(f"Суммарная награда за 10 шагов: {total_r:.2f}")