import yaml
from pathlib import Path

cfg_path = Path(__file__).parent / "config.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("Секции конфига:", list(cfg.keys()))
print("Горизонт:", cfg["simulation"]["horizon"])
print("Фазы:", cfg["topology"]["phase_names"])
print("λ по подходам:", cfg["traffic"]["arrival_rate"])
print("c_switch:", cfg["signal"]["c_switch"])
print("PPO total_timesteps:", cfg["rl"]["total_timesteps"])
print("Число seed-ов:", len(cfg["simulation"]["seeds"]))
print("OK")