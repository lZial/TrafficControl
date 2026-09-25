from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str], env: dict | None = None, desc: str = "") -> None:
    print("=" * 78)
    print(f">>> {desc}")
    print("=" * 78)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    print()
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПРОЕКТА traffic_control")
    print("Включает: Fixed, Adaptive, PPO (обучение + инференс), сравнения.")
    print("Время: ~30–40 минут на CPU.")
    print()

    run(
        [PYTHON, "-c", "import numpy, pandas, scipy, simpy, gymnasium, stable_baselines3; print('ok')"],
        desc="Шаг 0. Проверка зависимостей",
    )

    run(
        [PYTHON, "experiments/train_ppo.py"],
        env={**os.environ, "TRAFFIC_CONFIG": "config_medium.yaml"},
        desc="Шаг 1. Обучение PPO с доменной рандомизацией (1M шагов, ~13 мин)",
    )

    run(
        [PYTHON, "experiments/extract_vecnorm.py"],
        desc="Шаг 2. Извлечение статистик VecNormalize в .npy",
    )

    for load in ["low", "medium", "high"]:
        cfg = f"config_{load}.yaml"
        env = {**os.environ, "TRAFFIC_CONFIG": cfg}

        run([PYTHON, "experiments/run_fixed.py"], env=env,
            desc=f"Шаг 3.{load}. Fixed-time на нагрузке {load} (30 эпизодов)")

        run([PYTHON, "experiments/run_adaptive.py"], env=env,
            desc=f"Шаг 4.{load}. Adaptive на нагрузке {load} (30 эпизодов)")

        run([PYTHON, "experiments/run_rl.py"], env=env,
            desc=f"Шаг 5.{load}. PPO на нагрузке {load} (30 эпизодов)")

    run([PYTHON, "experiments/compare.py"],
        desc="Шаг 6. Сравнение трёх стратегий при средней нагрузке")

    run([PYTHON, "experiments/compare_loads.py"],
        desc="Шаг 7. Сравнение по трём нагрузкам")

    run([PYTHON, "experiments/compare_models.py"],
        desc="Шаг 8. Сравнение Base PPO и PPO+DR")

    print()
    print("=" * 78)
    print("ГОТОВО")
    print("Результаты:")
    print("  data/raw/*.csv           — сырые логи эпизодов")
    print("  reports/tables/*.csv     — сводные таблицы")
    print("  reports/figures/*.png    — графики")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print()
        print(f"ОШИБКА на шаге: {e.cmd}")
        print(f"Код возврата: {e.returncode}")
        sys.exit(1)