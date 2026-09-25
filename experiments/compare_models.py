from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config


BASE_PPO_W = {
    "low": 2.787,
    "medium": 5.304,
    "high": 423.459,
}


def main():
    cfg = load_config()
    d = Path(cfg.paths.data_raw)

    print("=" * 90)
    print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ PPO: Base (без DR) vs DR (domain randomization)")
    print("=" * 90)
    print(f"{'Нагрузка':<10} {'Base PPO':<12} {'PPO+DR':<12} {'Δ':<12} {'Δ %':<10}")
    print("-" * 90)

    rows = []
    for load in ["low", "medium", "high"]:
        dr_file = d / f"{load}_rl.csv"
        if not dr_file.exists():
            print(f"{load:<10} (файл {dr_file} не найден)")
            continue
        W_dr = pd.read_csv(dr_file)["W"].mean()
        W_base = BASE_PPO_W[load]
        delta = W_dr - W_base
        pct = 100 * delta / W_base
        rows.append({
            "load": load,
            "base_ppo_W": W_base,
            "ppo_dr_W": W_dr,
            "delta_W": delta,
            "delta_pct": pct,
        })
        print(f"{load:<10} {W_base:<12.3f} {W_dr:<12.3f} {delta:+12.3f} {pct:+9.1f}%")

    out = Path(cfg.paths.tables) / "base_vs_dr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()