# План реализации: Macro + Micro архитектура

> Архитектура: `docs/system_architecture.md`
> Статус обновляется по мере выполнения.
> Формат: `- [ ]` не сделано / `- [x]` готово

---

## Фаза 1 — Микро-улучшения

> Цель: улучшить качество рекомендаций не ломая текущую систему.
> Зависимости: нет, можно начинать сразу.

### 1.1 IRT фильтр перед LinUCB

**Файл:** `services/retrieval/main.py`, `services/retrieval/selector.py`

- [ ] Написать функцию `compute_p_correct(mastery: float, irt_difficulty: float) -> float`
  - `θ = log(mastery / (1 - mastery))` с клампингом mastery в `[0.01, 0.99]`
  - `P = 1 / (1 + exp(-(θ - irt_difficulty)))`
- [ ] Написать функцию `get_difficulty_range(mode: str) -> tuple[float, float]`
  - `build` → `(0.60, 0.75)`
  - `consolidate` → `(0.75, 0.90)`
  - `test` → `(0.45, 0.60)`
  - default (нет плана) → `(0.60, 0.75)`
- [ ] Добавить IRT фильтр в `/recommend` перед LinUCB scoring
  - Фильтровать `tasks` по `P_correct` в нужном диапазоне
  - Fallback: если после фильтра список пустой — снять фильтр (не оставлять ученика без задания)
- [ ] Написать unit-тесты: `services/retrieval/tests/test_irt_filter.py`
  - тест: `compute_p_correct` корректно считает при разных mastery/difficulty
  - тест: фильтр оставляет только задания в диапазоне
  - тест: fallback работает при пустом пуле

### 1.2 Заполнить x[1..4] — mastery пре-реквизитов

**Файлы:** `services/retrieval/main.py`, `services/retrieval/clients.py`

- [ ] Добавить в `clients.py` метод `get_kc_prerequisites(http, kc_id) -> list[str]`
  - Вызов `GET /nodes/{kc_id}/prerequisites` из graph-сервиса
- [ ] В `/recommend`: после выбора KC запросить её пре-реквизиты
- [ ] Обновить `_build_context()` — принять `prereq_masteries: list[float]`
  - Заполнить `x[1..4]` первыми 4 пре-реквизитами (отсортировать по силе ребра)
  - Если пре-реквизитов меньше 4 — заполнить нулями
- [ ] Написать тест: контекстный вектор корректно заполняется при наличии/отсутствии пре-реквизитов

### 1.3 seen_tasks с TTL

**Файл:** `services/profile/main.py`, `services/profile/repository.py`

- [ ] В эндпоинте `GET /students/{id}/seen-tasks` добавить параметр `ttl_days: int = 30`
- [ ] Фильтровать interactions по `created_at > now() - ttl_days`
- [ ] В `services/retrieval/clients.py` передавать `ttl_days=30` при запросе seen_tasks
- [ ] Написать тест: задания старше TTL не попадают в список seen_tasks

### 1.4 Проверка и прогон тестов Фазы 1

- [ ] `pytest services/retrieval/tests/ -v` — все зелёные
- [ ] `pytest services/profile/tests/ -v` — все зелёные
- [ ] Запустить `make dev` + ручная проверка: `make play` или `curl` запрос к `/recommend`

---

## Фаза 2 — Интерфейс Macro ↔ Micro

> Цель: дать микро-уровню директивы от макро и научить его отчитываться.
> Зависимости: Фаза 1 завершена.

### 2.1 DB миграция

**Файл:** `migrations/versions/0007_macro_micro_interface.py`

- [ ] Расширить таблицу `learning_plans`:
  - `goal_type VARCHAR` — `"target_mastery"` | `"coverage"`
  - `mastery_threshold FLOAT DEFAULT 0.80`
  - `require_test BOOLEAN DEFAULT FALSE`
  - `coverage_variant VARCHAR` — `"count"` | `"mass"` | `"frontier"`
  - `task_budget INT` — для Режима 2
- [ ] Расширить таблицу `plan_steps`:
  - `difficulty_mode VARCHAR DEFAULT 'build'`
  - `tasks_budget INT DEFAULT 20`
  - `tasks_spent INT DEFAULT 0`
  - `status VARCHAR DEFAULT 'pending'` — `pending` | `in_progress` | `suspended` | `completed`
- [ ] Запустить миграцию: `alembic upgrade head`
- [ ] Проверить: `alembic current` показывает `0007`

### 2.2 Micro читает difficulty_mode из плана

**Файл:** `services/retrieval/main.py`

- [ ] В `_get_plan_priorities()` также возвращать `difficulty_mode` активного шага
- [ ] Передавать `difficulty_mode` в IRT фильтр (из 1.1)
- [ ] Если нет активного плана → `difficulty_mode = "build"`
- [ ] Написать тест: при разных difficulty_mode фильтр применяет разные диапазоны

### 2.3 Micro обновляет tasks_spent

**Файл:** `services/retrieval/main.py`

- [ ] После выдачи задания: `UPDATE plan_steps SET tasks_spent = tasks_spent + 1`
  - Только для активного шага плана (`status = 'in_progress'`)
  - Только если у ученика есть активный план
- [ ] Написать тест: tasks_spent инкрементируется после каждой рекомендации

### 2.4 Kafka топики

**Файл:** `docker-compose.yml`, новый `services/retrieval/kafka_producer.py`

- [ ] Добавить топики в Kafka: `micro_summaries`, `macro_directives`
- [ ] Создать `services/retrieval/kafka_producer.py` с функцией `publish_micro_summary(summary: dict)`
- [ ] Создать `services/macro/kafka_consumer.py` (заглушка — пока только consume + log)

### 2.5 MicroSummary — вычисление и отправка

**Файл:** `services/retrieval/main.py`

- [ ] Добавить функцию `compute_micro_summary(student_id, kc_id, db) -> dict`
  - Читает последние N записей из `bandit_log` для этой KC
  - Считает: `mastery_before`, `mastery_after`, `velocity`, `frustration_count`, `avg_score`, `hint_rate`, `irt_residual`, `tasks_spent`
- [ ] Добавить плановый триггер в `/recommend`:
  - Проверить условие: `tasks_spent % 15 == 0` ИЛИ `Δmastery ≥ 0.1`
  - При срабатывании: вычислить MicroSummary → опубликовать в Kafka
- [ ] Добавить внеплановый триггер OnFrustration:
  - `frustration_count ≥ 2 AND velocity ≈ 0` → публиковать MicroSummary сразу
- [ ] Написать тест: MicroSummary содержит корректные поля

### 2.6 Bridge bonus

**Файл:** `services/retrieval/main.py`

- [ ] В `_get_plan_priorities()` также возвращать `next_plan_kc_id` (следующий шаг)
- [ ] После LinUCB scoring: `if next_plan_kc in task.secondary_kcs: score += 0.1`
- [ ] Написать тест: задание-мостик получает бонус, non-bridge — нет

### 2.7 Проверка Фазы 2

- [ ] `pytest services/retrieval/tests/ -v`
- [ ] `pytest services/profile/tests/ -v`
- [ ] E2E проверка: создать план вручную в БД → запустить `/recommend` → убедиться что tasks_spent растёт и difficulty_mode применяется

---

## Фаза 3 — Macro Planner (ядро)

> Цель: новый сервис `services/macro/` — строит планы и реагирует на события.
> Зависимости: Фаза 2 завершена.

### 3.1 Новый сервис services/macro/

- [ ] Создать структуру:
  ```
  services/macro/
    __init__.py
    main.py            — FastAPI: POST /plans, GET /plans/{id}, POST /plans/{id}/evaluate
    prereq_extractor.py
    bkt_environment.py
    tasks_estimator.py
    policy_mode1.py
    plan_lifecycle.py
    kafka_consumer.py
    tests/
      __init__.py
      test_prereq_extractor.py
      test_bkt_environment.py
      test_tasks_estimator.py
      test_plan_lifecycle.py
  ```
- [ ] Добавить в `Makefile` и `docker-compose.yml`
- [ ] Добавить порт: `macro=8006`

### 3.2 PrereqSubgraphExtractor

**Файл:** `services/macro/prereq_extractor.py`

- [ ] Функция `extract_prereq_subgraph(target_kc_id, mastery, threshold=0.75) -> dict`
  - BFS назад по Neo4j от `target_kc_id`
  - Фильтр: включать KC где `mastery[kc] < threshold - 0.05`
  - Возвращать: `{nodes: [...], edges: [...]}` с mastery и edge_strength
- [ ] Unit-тесты: корректный BFS, правильная фильтрация по mastery

### 3.3 BKTEnvironment (симулятор для RL)

**Файл:** `services/macro/bkt_environment.py`

- [ ] Класс `BKTEnvironment`:
  - `reset(initial_mastery: dict, subgraph: dict) -> state`
  - `step(kc_id: str) -> (new_state, reward, done)`
    - Симулировать N микро-шагов через `bkt.update_mastery()`
    - Reward: `Δmastery(target) + 0.1 × Σ Δmastery(prereq) × edge_strength − 0.01`
    - Done: `mastery[target_kc] ≥ target_mastery`
- [ ] Unit-тесты: симулятор корректно обновляет mastery, reward считается правильно

### 3.4 TasksToMasteryEstimator

**Файл:** `services/macro/tasks_estimator.py`

- [ ] Аналитическая оценка: `simulate_tasks_to_mastery(bkt_params, m_current, m_target, n_sims=500)`
  - Запустить BKT симуляцию N раз, вернуть медиану
- [ ] Гибридная оценка: `estimate(kc, m_current, m_target, student_id, cluster_id) -> int`
  - `α = min(1.0, n_practiced_in_subject / 20)`
  - `α × cluster_avg + (1-α) × simulation_estimate`
- [ ] Unit-тесты: оценка растёт с увеличением mastery-gap, уменьшается с ростом p_transit

### 3.5 TargetMasteryPolicy (RL, Режим 1)

**Файл:** `services/macro/policy_mode1.py`

- [ ] Класс `SubgraphQAgent`:
  - Q-таблица: `state_key → {kc_id: q_value}`
  - `select_action(state, subgraph, epsilon=0.1) -> kc_id`
  - `update(state, action, reward, next_state, gamma=0.9)`
- [ ] Функция `train_policy(cluster_id, target_kc, n_episodes=5000) -> SubgraphQAgent`
  - Инициализация из кластерного центроида
  - Цикл: `env.reset() → select_action → env.step → agent.update`
  - Сохранять Q-таблицу в файл: `models/policy_cluster_{id}_{target_kc}.pkl`
- [ ] CLI скрипт: `python -m services.macro.policy_mode1 --cluster 0 --target quadratic_eq`
- [ ] Unit-тесты: агент обучается (Q-values сходятся), выбирает действия корректно

### 3.6 PlanLifecycleManager

**Файл:** `services/macro/plan_lifecycle.py`

- [ ] `create_plan(student_id, mode, params) -> plan_id`
  - Режим 1: вызвать `PrereqSubgraphExtractor` → загрузить политику кластера → развернуть план
  - Записать шаги в `plan_steps` с `tasks_budget` от `TasksToMasteryEstimator`
- [ ] `evaluate_micro_summary(summary: dict) -> list[PlanAction]`
  - Уровень 1: `difficulty_mode = consolidate`
  - Уровень 2: вставить слабый prereq как новый `plan_step`
  - Уровень 3: вызвать `create_plan` заново от текущего mastery
- [ ] `advance_step(plan_id, student_id)` — отметить текущий шаг completed, активировать следующий
- [ ] `check_test_phase(plan_id, current_mastery) -> bool` — нужно ли переключить на `test` mode
- [ ] Kafka consumer: слушать `micro_summaries` → вызывать `evaluate_micro_summary`
- [ ] Unit-тесты: все три уровня эскалации, переход между шагами, test-фаза

### 3.7 FastAPI эндпоинты macro-сервиса

**Файл:** `services/macro/main.py`

- [ ] `POST /plans` — создать план (тело: `PlanRequest`)
- [ ] `GET /plans/{plan_id}` — статус плана и текущий шаг
- [ ] `POST /plans/{plan_id}/evaluate` — ручной trigger пересмотра (для тестов)
- [ ] `GET /health`
- [ ] Добавить вызов `POST /plans` в `services/gateway/clients.py`

### 3.8 Проверка Фазы 3

- [ ] `pytest services/macro/tests/ -v`
- [ ] E2E тест:
  - Создать ученика → создать план (Режим 1) → решить 30 заданий → убедиться что план продвигается
  - Симулировать фрустрацию → убедиться что уровень 2 вставляет prereq
- [ ] Запустить симуляцию: `python tools/simulate.py` — все сценарии зелёные

---

## Фаза 4 — Coverage Policy (Режим 2)

> Цель: второй режим планирования — максимальное покрытие за N задач.
> Зависимости: Фаза 3 завершена.

### 4.1 LearningSpeedModel

**Файл:** `services/macro/learning_speed.py`

- [ ] `estimate_speed(kc_id, student_id, cluster_id) -> float`
  - Читать `cluster_task_stats` → считать avg Δmastery/task для subject + difficulty_bin
  - Читать личную историю студента из `bandit_log`
  - Смешивать с весом `α`
- [ ] Unit-тест: при большой личной истории — вес cluster_prior падает

### 4.2 CoveragePolicy (RL с бюджетом)

**Файл:** `services/macro/policy_mode2.py`

- [ ] Класс `CoverageQAgent`:
  - State: `(mastery_summary, tasks_remaining_bin)` — дискретизировать для Q-таблицы
  - Action: выбрать KC из ZPD
  - Reward: зависит от `coverage_variant` (`count` / `mass` / `frontier`)
- [ ] `get_reward(kc_id, mastery_before, mastery_after, variant) -> float`
  - `count`: `+1.0` если mastery пересекла порог, иначе `0`
  - `mass`: `mastery_after - mastery_before`
  - `frontier`: `(mastery_after - mastery_before) × (2.0 если kc раньше не практиковалась, иначе 1.0)`
- [ ] Функция `train_coverage_policy(cluster_id, variant, n_episodes=5000) -> CoverageQAgent`
- [ ] Unit-тесты: разные варианты reward дают разные политики (count агент дожимает KC до порога, frontier агент исследует новые KC)

### 4.3 Интеграция Режима 2 в PlanLifecycleManager

**Файл:** `services/macro/plan_lifecycle.py`

- [ ] В `create_plan()` обработать `mode="coverage"`:
  - Загрузить `CoveragePolicy` нужного кластера и варианта
  - Создать "динамический план" — не список KC, а политика + бюджет
  - Записать в `learning_plans` с `goal_type="coverage"`, `task_budget`
- [ ] Обновить `evaluate_micro_summary()` для coverage-режима:
  - `OnBudgetAlert`: если `tasks_remaining < 20%` → приоритизировать KC близкие к порогу

### 4.4 Проверка Фазы 4

- [ ] `pytest services/macro/tests/ -v`
- [ ] E2E тест Режима 2:
  - Создать ученика → создать coverage-план (100 задач, variant=count) → решить задания → убедиться что coverage растёт
  - Проверить все три варианта: count, mass, frontier
- [ ] Сравнительный тест: coverage при Режиме 2 > coverage при свободном режиме за то же число задач

---

## Итоговая проверка системы

- [ ] Запустить полный тест-сьют: `pytest -v` — все тесты зелёные
- [ ] Запустить симуляцию `tools/simulate.py` — все 9 сценариев проходят
- [ ] Smoke-тест полного флоу:
  - Создать ученика grade=8
  - Создать план Режим 1 (цель: тема алгебра grade 9)
  - Решить 50 заданий через API
  - Убедиться: план продвигается, difficulty_mode меняется, MicroSummary публикуется
- [ ] Проверить latency горячего пути: `P95 < 500ms` (с учётом новых запросов к graph)

---

## Отложено (не в этом плане)

- [ ] CollaborativePlanner — нужны реальные данные ≥ 200 учеников
- [ ] Онлайн RL (дообучение политики в реальном времени) — после накопления данных
- [ ] Полная IRT калибровка (параметры a, c) — нужны данные
- [ ] Theta per KC в профиле — перспектива после накопления данных
