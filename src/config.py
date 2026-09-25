from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

import os

def _default_config_path() -> Path:
    env = os.environ.get("TRAFFIC_CONFIG")
    if env:
        return Path(env).resolve()
    import os

def _default_config_path() -> Path:
    env = os.environ.get("TRAFFIC_CONFIG")
    if env:
        return (Path(__file__).resolve().parent.parent / env).resolve()
    return Path(__file__).resolve().parent.parent / "config_medium.yaml"

CONFIG_PATH = _default_config_path()

CONFIG_PATH = _default_config_path()


@dataclass
class SimulationCfg:
    dt: float
    horizon: int
    n_episodes: int
    seeds: List[int]


@dataclass
class TopologyCfg:
    n_intersections: int
    approaches: List[str]
    n_phases: int
    phase_names: List[str]
    phase_green: Dict[str, List[str]]


@dataclass
class TrafficCfg:
    arrival_rate: Dict[str, float]
    service_rate: float


@dataclass
class SignalCfg:
    min_green: int
    max_green: int
    initial_phase: str
    c_switch: float


@dataclass
class NormalizationCfg:
    max_queue: float
    max_wait: float


@dataclass
class RLCfg:
    algorithm: str
    policy: str
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    verbose: int
    model_path: str
    tensorboard_log: str


@dataclass
class AdaptiveCfg:
    queue_low_threshold: int
    queue_high_threshold: int


@dataclass
class FixedCfg:
    lost_time_per_phase: float
    saturation_flow: float
    min_cycle: int
    max_cycle: int


@dataclass
class MetricsCfg:
    names: List[str]


@dataclass
class StatsCfg:
    confidence: float
    test: str
    effect_size: str


@dataclass
class PathsCfg:
    data_raw: str
    data_processed: str
    models: str
    figures: str
    tables: str


@dataclass
class Config:
    simulation: SimulationCfg
    topology: TopologyCfg
    traffic: TrafficCfg
    signal: SignalCfg
    normalization: NormalizationCfg
    rl: RLCfg
    adaptive: AdaptiveCfg
    fixed: FixedCfg
    metrics: MetricsCfg
    stats: StatsCfg
    paths: PathsCfg


def _build(cls, d):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in d.items() if k in fields})


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        simulation=_build(SimulationCfg, raw["simulation"]),
        topology=_build(TopologyCfg, raw["topology"]),
        traffic=_build(TrafficCfg, raw["traffic"]),
        signal=_build(SignalCfg, raw["signal"]),
        normalization=_build(NormalizationCfg, raw["normalization"]),
        rl=_build(RLCfg, raw["rl"]),
        adaptive=_build(AdaptiveCfg, raw["adaptive"]),
        fixed=_build(FixedCfg, raw["fixed"]),
        metrics=_build(MetricsCfg, raw["metrics"]),
        stats=_build(StatsCfg, raw["stats"]),
        paths=_build(PathsCfg, raw["paths"]),
    )


if __name__ == "__main__":
    cfg = load_config()
    print("horizon =", cfg.simulation.horizon)
    print("approaches =", cfg.topology.approaches)
    print("c_switch =", cfg.signal.c_switch)
    print("OK")