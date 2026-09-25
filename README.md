Модель интеллектуального управления дорожным трафиком

Сравнение трёх стратегий управления светофором на микроскопической
модели транспортного потока на языке Python.
Стратегии

1. Fixed-time - фиксированный цикл по формуле Вебстера.
2. Adaptive - rule-based управление по порогам очередей.
3. Intelligent (RL) - PPO из `stable-baselines3` с доменной
   рандомизацией нагрузки.

Установка

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install tensorboard rich

Открытие папки проекта и активация окружения:
cd /d D:\traffic_control
.venv\Scripts\activate

Запуск в одну команду

Полный пайплайн (обучение PPO + все прогоны + сравнения) одной командой:

Полная установка с нуля и запуск:
    setup_and_run.bat

Windows:
    run_all.bat

Linux / macOS / Windows (вручную):
    python run_all.py

Время выполнения: ~30–40 минут на CPU.

Скрипт делает:
1. Проверяет зависимости.
2. Обучает PPO с доменной рандомизацией (1M шагов).
3. Извлекает статистики VecNormalize.
4. Прогоняет три стратегии на трёх уровнях нагрузки (9 × 30 эпизодов).
5. Строит сводные таблицы и графики.

Конфигурации

В проекте три конфигурации с разной интенсивностью потока:

- `config_low.yaml`    - низкая нагрузка (~60% от пропускной способности)
- `config_medium.yaml` - средняя нагрузка (~80%), используется по умолчанию
- `config_high.yaml`   - высокая нагрузка (~96%)

Выбор конфигурации - через переменную окружения `TRAFFIC_CONFIG`:

set TRAFFIC_CONFIG=config_low.yaml

Если переменная не установлена, используется `config_medium.yaml`.

Запуск экспериментов

1. Прогон Fixed и Adaptive на одной нагрузке

set TRAFFIC_CONFIG=config_medium.yaml
python experiments/run_fixed.py
python experiments/run_adaptive.py

Результаты сохраняются в `data/raw/medium_fixed.csv` и
`data/raw/medium_adaptive.csv`.

2. Обучение PPO с доменной рандомизацией

python experiments/train_ppo.py

Обучение занимает 12–20 минут на CPU. Модель сохраняется в
`models/ppo_traffic.zip`, статистики `VecNormalize` - в
`models/vec_normalize.pkl`.

3. Прогон обученного агента

python experiments/extract_vecnorm.py

set TRAFFIC_CONFIG=config_low.yaml
python experiments/run_rl.py

set TRAFFIC_CONFIG=config_medium.yaml
python experiments/run_rl.py

set TRAFFIC_CONFIG=config_high.yaml
python experiments/run_rl.py

Результаты - в `data/raw/{low,medium,high}_rl.csv`.

4. Сводные сравнения

python experiments/compare.py
python experiments/compare_loads.py
python experiments/compare_models.py

Скрипты формируют таблицы в `reports/tables/` и графики в
`reports/figures/`.

Структура проекта

- `src/` - код среды, стратегий и метрик
- `src/envs/traffic_env.py` - gymnasium-обёртка для PPO
- `src/strategies/` - Fixed, Adaptive, RL
- `experiments/` - скрипты запуска, обучения и сравнения
- `config_low.yaml`, `config_medium.yaml`, `config_high.yaml` - три режима нагрузки
- `data/raw/` - сырые логи эпизодов (CSV)
- `data/processed/` - агрегаты для отчёта
- `models/` - сохранённые модели PPO и статистики VecNormalize
- `reports/figures/` - графики для отчёта
- `reports/tables/` - таблицы для отчёта

Ключевые результаты

Метрика - среднее время ожидания W (с), 30 эпизодов на конфигурацию.

| Нагрузка | Fixed | Adaptive | PPO+DR |
|----------|-------|----------|--------|
| Low      | 8.85  | 4.80     | 4.37   |
| Medium   | 20.94 | 7.44     | 8.19   |
| High     | 52.52 | 21.75    | 27.91  |

Сравнение двух обученных политик PPO:

| Нагрузка | Base PPO | PPO+DR | Изменение |
|----------|----------|--------|-----------|
| Low      | 2.79     | 4.37   | +56.6%    |
| Medium   | 5.30     | 8.19   | +54.4%    |
| High     | 423.46   | 27.91  | -93.4%    |

Воспроизводимость

Все эпизоды воспроизводимы по seed-ам из `config_*.yaml`.
Список seeds: 0..29. Симулятор детерминирован при фиксированном seed.
Обучение PPO также воспроизводимо при фиксированном `seed=0`.