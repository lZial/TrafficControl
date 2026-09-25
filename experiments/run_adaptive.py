from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.simulator import TrafficSimulator
from src.strategies.adaptive import AdaptiveStrategy


def run_one_episode(cfg, seed: int) -> dict:
    sim = TrafficSimulator(cfg, seed=seed)
    strategy = AdaptiveStrategy(cfg)
    strategy.reset()
    summary = sim.run(strategy.as_controller(), horizon=cfg.simulation.horizon)
    summary["strategy"] = strategy.name
    summary["q_low"] = strategy.q_low
    summary["q_high"] = strategy.q_high
    return summary


def main() -> None:
    cfg = load_config()

    probe = AdaptiveStrategy(cfg)
    print(probe.describe())

    rows = []
    for seed in tqdm(cfg.simulation.seeds, desc="Adaptive episodes"):
        rows.append(run_one_episode(cfg, seed))

    df = pd.DataFrame(rows)

    metric_cols = ["W", "Lmax", "throughput", "total_delay"]
    print("\n=== Adaptive: средние за 30 эпизодов ===")
    print(df[metric_cols].agg(["mean", "std"]).round(4))

    import os
    cfg_env = os.environ.get("TRAFFIC_CONFIG", "config.yaml")
    tag = Path(cfg_env).stem.replace("config_", "").replace("config",     "medium")
    out_dir = Path(cfg.paths.data_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_adaptive.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()