"""Проверка симулятора без стратегии: ручное переключение фазы каждые 30 с."""
from src.config import load_config
from src.simulator import TrafficSimulator


def dumb_controller(inter, env):
    # переключаем каждые 30 секунд, если можно
    if inter.t_phase >= 30 and inter.can_switch():
        return 1
    return 0


if __name__ == "__main__":
    cfg = load_config()
    sim = TrafficSimulator(cfg, seed=0)
    summary = sim.run(dumb_controller, horizon=600)  # 10 минут для быстрой проверки
    print("Метрики за 10 минут:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # первые 5 снимков
    print("\nПервые 5 шагов:")
    for snap in sim.history[:5]:
        print(snap)
    print("\nПоследние 5 шагов:")
    for snap in sim.history[-5:]:
        print(snap)