from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.config import load_config
from src.envs.traffic_env import TrafficEnv


def main():
    cfg = load_config()
    vn_path = str(Path(cfg.paths.models) / "vec_normalize.pkl")

    # создаём dummy-среду только чтобы загрузить VecNormalize — не шагаем по ней
    dummy = DummyVecEnv([lambda: TrafficEnv(cfg=cfg, seed=0)])
    vn = VecNormalize.load(vn_path, dummy)

    print("obs_rms.mean:", vn.obs_rms.mean)
    print("obs_rms.var :", vn.obs_rms.var)
    print("ret_rms.mean:", vn.ret_rms.mean)

    np.save(Path(cfg.paths.models) / "obs_mean.npy", vn.obs_rms.mean)
    np.save(Path(cfg.paths.models) / "obs_var.npy", vn.obs_rms.var)
    np.save(Path(cfg.paths.models) / "ret_rms.npy", vn.ret_rms.mean)
    print("Сохранено в models/*.npy")

    dummy.close()


if __name__ == "__main__":
    main()