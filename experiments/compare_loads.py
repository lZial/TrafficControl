from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config


LOADS = ["low", "medium", "high"]
STRATEGIES = ["fixed", "adaptive", "rl"]
COLORS = {"fixed": "#888888", "adaptive": "#3b82f6", "rl": "#ef4444"}
LABELS = {"fixed": "Fixed", "adaptive": "Adaptive", "rl": "RL (PPO)"}
METRICS = ["W", "Lmax", "throughput", "total_delay"]


def load_all(cfg):
    d = Path(cfg.paths.data_raw)
    data = {}
    for load in LOADS:
        for s in STRATEGIES:
            p = d / f"{load}_{s}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Не найден {p}")
            data[(load, s)] = pd.read_csv(p)
    return data


def main():
    cfg = load_config()
    data = load_all(cfg)

    print("=" * 90)
    print("СВОДНАЯ ТАБЛИЦА: W (среднее время ожидания), 30 эпизодов на комбинацию")
    print("=" * 90)
    rows = []
    for load in LOADS:
        row = {"load": load}
        for s in STRATEGIES:
            df = data[(load, s)]
            row[f"{s}_mean"] = df["W"].mean()
            row[f"{s}_std"] = df["W"].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))
    print()

    print("=" * 90)
    print("СВОДНАЯ ТАБЛИЦА: total_delay (суммарная задержка), 30 эпизодов")
    print("=" * 90)
    rows = []
    for load in LOADS:
        row = {"load": load}
        for s in STRATEGIES:
            df = data[(load, s)]
            row[f"{s}_mean"] = df["total_delay"].mean()
        rows.append(row)
    summary_td = pd.DataFrame(rows)
    print(summary_td.round(1).to_string(index=False))
    print()

    out_t = Path(cfg.paths.tables)
    out_f = Path(cfg.paths.figures)
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_t / "loads_summary_W.csv", index=False, encoding="utf-8")
    summary_td.to_csv(out_t / "loads_summary_total_delay.csv", index=False, encoding="utf-8")
    print(f"Таблицы сохранены в {out_t}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(LOADS))
    width = 0.26

    for ax, metric in zip(axes.ravel(), METRICS):
        for i, s in enumerate(STRATEGIES):
            means = [data[(load, s)][metric].mean() for load in LOADS]
            stds = [data[(load, s)][metric].std() for load in LOADS]
            ax.bar(
                x + (i - 1) * width, means, width,
                yerr=stds, capsize=4,
                label=LABELS[s],
                color=COLORS[s],
                edgecolor="black", linewidth=0.7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Low (~60%)", "Medium (~80%)", "High (~96%)"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Три стратегии при трёх уровнях нагрузки (30 эпизодов, среднее ± std)", fontsize=12)
    fig.tight_layout()
    out_png = out_f / "loads_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"График сохранён: {out_png}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for s in STRATEGIES:
        means = [data[(load, s)]["W"].mean() for load in LOADS]
        errs = [data[(load, s)]["W"].std() for load in LOADS]
        ax2.errorbar(x, means, yerr=errs, marker="o", markersize=8, capsize=6,
                     linewidth=2, label=LABELS[s], color=COLORS[s])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low (~60%)", "Medium (~80%)", "High (~96%)"])
    ax2.set_ylabel("Среднее время ожидания W, с")
    ax2.set_title("Зависимость W от уровня нагрузки")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out_png2 = out_f / "W_vs_load.png"
    fig2.savefig(out_png2, dpi=150)
    print(f"График сохранён: {out_png2}")


if __name__ == "__main__":
    main()