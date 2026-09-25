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
from scipy import stats

from src.config import load_config


def load_all(cfg):
    d = Path(cfg.paths.data_raw)
    dfs = {}
    for name in ["fixed", "adaptive", "rl"]:
        p = d / f"medium_{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Не найден {p}. Сначала запусти experiments/run_{name}.py")
        dfs[name] = pd.read_csv(p)
    return dfs


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - tcrit * se, mean + tcrit * se


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2)
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def welch_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def main():
    cfg = load_config()
    dfs = load_all(cfg)

    metrics = ["W", "Lmax", "throughput", "total_delay"]
    strategies = ["fixed", "adaptive", "rl"]

    print("=" * 78)
    print("СВОДНОЕ СРАВНЕНИЕ ТРЁХ СТРАТЕГИЙ")
    print("=" * 78)
    print(f"n = {len(dfs['fixed'])} эпизодов на стратегию")
    print()

    table_rows = []
    for m in metrics:
        row = {"metric": m}
        for s in strategies:
            mean, lo, hi = ci95(dfs[s][m])
            row[f"{s}_mean"] = mean
            row[f"{s}_ci_lo"] = lo
            row[f"{s}_ci_hi"] = hi
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    print(table.round(3).to_string(index=False))
    print()

    print("=" * 78)
    print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ (t-тест Уэлча, α = 0.05)")
    print("=" * 78)
    comparisons = [
        ("rl", "adaptive"),
        ("rl", "fixed"),
        ("adaptive", "fixed"),
    ]
    sig_rows = []
    for m in metrics:
        for a, b in comparisons:
            t, p = welch_t(dfs[a][m], dfs[b][m])
            d = cohen_d(dfs[a][m], dfs[b][m])
            sig = "ДА" if p < 0.05 else "нет"
            sig_rows.append({
                "metric": m,
                "A": a,
                "B": b,
                "mean_A": dfs[a][m].mean(),
                "mean_B": dfs[b][m].mean(),
                "t": t,
                "p": p,
                "significant": sig,
                "cohen_d": d,
            })
    sig_df = pd.DataFrame(sig_rows)
    print(sig_df.round(4).to_string(index=False))
    print()

    out_t = Path(cfg.paths.tables)
    out_f = Path(cfg.paths.figures)
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_t / "summary_table.csv", index=False, encoding="utf-8")
    sig_df.to_csv(out_t / "significance.csv", index=False, encoding="utf-8")
    print(f"Таблицы сохранены в {out_t}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"fixed": "#888888", "adaptive": "#3b82f6", "rl": "#ef4444"}
    titles = {
        "W": "Среднее время ожидания W, с",
        "Lmax": "Максимальная длина очереди Lmax",
        "throughput": "Пропускная способность, авто/с",
        "total_delay": "Суммарная задержка, авто·с",
    }
    for ax, m in zip(axes.ravel(), metrics):
        means = []
        los = []
        his = []
        for s in strategies:
            mu, lo, hi = ci95(dfs[s][m])
            means.append(mu)
            los.append(mu - lo)
            his.append(hi - mu)
        xs = np.arange(len(strategies))
        bars = ax.bar(
            xs, means, yerr=[los, his], capsize=6,
            color=[colors[s] for s in strategies],
            edgecolor="black", linewidth=0.8,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(["Fixed", "Adaptive", "RL (PPO)"])
        ax.set_title(titles[m])
        ax.grid(axis="y", alpha=0.3)
        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mu:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Сравнение трёх стратегий управления светофором (30 эпизодов, 95% ДИ)", fontsize=12)
    fig.tight_layout()
    out_png = out_f / "comparison_bars.png"
    fig.savefig(out_png, dpi=150)
    print(f"График сохранён: {out_png}")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    data = [dfs[s]["W"].values for s in strategies]
    bp = ax2.boxplot(data, tick_labels=["Fixed", "Adaptive", "RL (PPO)"], patch_artist=True)
    for patch, s in zip(bp["boxes"], strategies):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.6)
    ax2.set_ylabel("W, с")
    ax2.set_title("Распределение среднего времени ожидания W по 30 эпизодам")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_box = out_f / "W_boxplot.png"
    fig2.savefig(out_box, dpi=150)
    print(f"Ящик с усами сохранён: {out_box}")


if __name__ == "__main__":
    main()