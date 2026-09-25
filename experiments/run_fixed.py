from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.simulator import TrafficSimulator
from src.strategies.fixed import FixedStrategy


def run_one_episode(cfg, seed: int) -> dict:
    sim = TrafficSimulator(cfg, seed=seed)
    strategy = FixedStrategy(cfg)
    strategy.reset()
    summary = sim.run(strategy.as_controller(), horizon=cfg.simulation.horizon)
    summary["strategy"] = strategy.name
    summary["cycle"] = strategy.cycle
    summary["g_NS"] = strategy.green.get("NS", 0.0)
    summary["g_EW"] = strategy.green.get("EW", 0.0)
    return summary


def main() -> None:
    cfg = load_config()

    probe = FixedStrategy(cfg)
    print(probe.describe())

    rows = []
    for seed in tqdm(cfg.simulation.seeds, desc="Fixed episodes"):
        rows.append(run_one_episode(cfg, seed))

    df = pd.DataFrame(rows)

    metric_cols = ["W", "Lmax", "throughput", "total_delay"]
    print("\n=== Fixed-time: средние за 30 эпизодов ===")
    print(df[metric_cols].agg(["mean", "std"]).round(4))

    cfg_env = os.environ.get("TRAFFIC_CONFIG", "config.yaml")
    tag = Path(cfg_env).stem.replace("config_", "").replace("config", "medium")
    out_dir = Path(cfg.paths.data_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_fixed.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()