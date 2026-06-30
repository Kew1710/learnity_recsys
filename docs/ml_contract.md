# ML Contract: Learnity Recommendation System

## 1. System Objective

**Primary online objective:**
Maximize learning progress under acceptable frustration.

**Secondary objectives:**
- Не зацикливаться на одной KC (cooldown, subject rotation).
- Не выдавать задания за пределами ZPD.
- Различать контентный дефицит и проблему ученика.
- Сохранять объяснимость каждого решения (decision log).

---

## 2. Student State Contract

### 2.1 Текущие обязательные поля (production)

| Поле | Источник | Тип | Описание |
|------|----------|-----|----------|
| `probability` | `mastery.probability` | float [0,1] | EMA-оценка знания KC |
| `probability_effective` | computed | float [0,1] | `probability` после decay |
| `attempts_count` | `mastery.attempts_count` | int | Количество попыток по KC |
| `consecutive_errors` | `mastery.consecutive_errors` | int | Ошибки подряд по KC |
| `consecutive_correct` | `mastery.consecutive_correct` | int | Правильные ответы подряд по KC |
| `grade` | `students.grade` | int | Класс ученика |
| `guessing_rate` | `students.guessing_rate` | float | Склонность к угадыванию (NOT UPDATED) |
| `hint_dependence` | `students.hint_dependence` | float | Зависимость от подсказок (NOT UPDATED) |
| `estimated_lr` | `students.estimated_lr` | float | Индивидуальная скорость обучения EMA |
| `review_mode` | `students.review_mode` | bool | Режим повторения (без decay) |
| `cluster_id` | `student_clusters.cluster_id` | int \| null | Кластер ученика |

### 2.2 Добавляемые поля (Пакет 1)

| Поле | Назначение | Формула / источник |
|------|------------|-------------------|
| `confidence` | Уверенность в оценке mastery | `f(attempts_count, recency, stability)` — точная формула ниже |
| `guessing_rate` (update) | Начать обновлять из поведения | EMA от `score` при `hints_used == 0 AND irt_p_correct < 0.3` |
| `hint_dependence` (update) | Начать обновлять из поведения | EMA от `hints_used > 0` rate за последние N заданий |

#### Формула `confidence`

```
stability = 1 - abs(recent_accuracy - probability_effective)
recency = 0.5 ^ (days_since_last_practiced / 30)
confidence = min(1.0, (attempts_count / 10) * stability * recency)
```

Интерпретация:
- `confidence < 0.3` — не принимать жёстких решений (не завершать шаг плана, не переводить в completed).
- `confidence >= 0.7` — можно доверять mastery для ZPD и plan thresholds.

### 2.3 Поля для логирования (для будущих моделей)

Эти поля записываются в `bandit_log` / `interactions`, но не влияют на текущие решения:

| Поле | Назначение |
|------|------------|
| `sequence_position` | Порядковый номер ответа ученика (глобальный) |
| `kc_sequence_position` | Порядковый номер ответа по конкретной KC |
| `mastery_snapshot_before` | Mastery до обновления |
| `mastery_snapshot_after` | Mastery после обновления |
| `mode_at_decision` | `build / consolidate / test / diagnostic` |

---

## 3. Task State Contract

### 3.1 Текущие поля задания (из TaskBank)

| Поле | Тип | Описание |
|------|-----|----------|
| `task_id` | UUID | Уникальный идентификатор |
| `grade_min` | int | Минимальный класс |
| `parts[].primary_kcs` | list[str] | KC, которые тестирует часть |
| `parts[].secondary_kcs` | list[str] | Связанные KC |
| `parts[].irt.difficulty` | float | IRT-сложность (b-параметр) |
| `parts[].irt.discrimination` | float | IRT-различение (a-параметр) |
| `parts[].irt.guessing` | float | IRT-угадывание (c-параметр) |
| `parts[].answer_type` | str | `numeric / multiple_choice / multi_select / proof` |
| `parts[].scaffolding_steps` | list[str] | Пошаговая подсказка |
| `parts[].distractors_map` | dict | Ответ -> misconception_id |
| `parts[].task_type` | str | `procedural / conceptual / word_problem / mixed` |
| `parts[].n_steps` | int | Количество шагов решения |

### 3.2 Вычисляемые метрики (Пакет 1-2)

| Поле | Назначение | Источник |
|------|------------|----------|
| `success_rate` | Доля правильных ответов по заданию | Агрегат из `bandit_log` |
| `coverage_count_by_kc` | Сколько заданий доступно для KC | Count из TaskBank |
| `coverage_by_difficulty_band` | Покрытие по полосам сложности для KC | Count из TaskBank per difficulty quartile |

---

## 4. Reward Contract

### 4.1 Текущий surrogate reward (production)

```
reward = delta_mastery - 0.3 * frustration - 0.1 * boredom
```

Где:
- `delta_mastery = mean(mastery_after[kc] - mastery_before[kc])` для primary_kcs
- `frustration = 1.0` если consecutive_errors >= 3 для любой primary KC, иначе 0.0
- `boredom = 1.0` если score >= 0.5 AND mastery_before > 0.9 для любой primary KC, иначе 0.0

**Известные проблемы:**
- `delta_mastery` вычисляется через EMA (артефакт модели, не реальный прогресс).
- Нет внешнего сигнала (pre/post-test, IRT-calibrated P(correct)).
- LinUCB оптимизирует артефакт собственной knowledge model.

### 4.2 Контракт на расширение reward (Пакет 2+)

Вводится отдельный `observed_outcome_features` dict, который логируется рядом с reward:

| Feature | Описание |
|---------|----------|
| `delta_mastery` | Как сейчас |
| `frustration` | Как сейчас |
| `boredom` | Как сейчас |
| `confidence_before` | Confidence mastery до ответа |
| `confidence_after` | Confidence mastery после ответа |
| `irt_residual` | `abs(P_correct_irt - actual_score)` |
| `diagnosed_reason` | Причина результата (prereq_gap / content_gap / ...) |

Текущий `reward` остаётся как surrogate. Будущий reward будет строиться как weighted combination из `observed_outcome_features`, калибруемая offline.

---

## 5. Decision Log Contract

### 5.1 Текущая структура `bandit_log`

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | UUID | PK |
| `student_id` | UUID | Ученик |
| `task_id` | UUID | Выданное задание |
| `kc_id` | str | KC задания |
| `context_vector` | float[13] | Контекстный вектор LinUCB |
| `reward` | float \| null | Заполняется после ответа |
| `recommended_at` | datetime | Момент рекомендации |
| `answered_at` | datetime \| null | Момент ответа |

### 5.2 Добавляемые поля (Пакет 1)

| Поле | Тип | Описание |
|------|-----|----------|
| `selection_reason` | str | Почему выбрана эта KC: `plan_lock / zpd_rotation / cooldown_fallback / subject_rotation` |
| `exploration_type` | str | Тип exploration: `exploit / cluster_explore / epsilon_greedy / phase1_heuristic / control_heuristic` |
| `zpd_candidates_count` | int | Сколько KC было в ZPD при выборе |
| `plan_step_id` | UUID \| null | ID активного шага плана (если есть) |
| `difficulty_mode` | str | Режим обучения: `build / consolidate / test` |
| `fallback_occurred` | bool | True если plan KC не имел заданий и произошёл fallback |

Эти поля позволяют:
1. Ответить "почему система выдала именно это задание".
2. Отследить частоту fallback с plan KC (контентный дефицит).
3. Построить offline replay: воспроизвести decision path без системы.

---

## 6. Context Vector Contract (LinUCB / Thompson Sampling)

### 6.1 Текущий 13-dim вектор

| Индекс | Поле | Формула |
|--------|------|---------|
| 0 | `mastery_kc` | `probability_effective` текущей KC |
| 1-4 | `prereq_masteries` | Top-4 mastery prereqs по силе связи, padding 0 |
| 5 | `errors_streak` | `consecutive_errors` для KC |
| 6 | `grade_norm` | `grade / 11.0` |
| 7 | `avg_reward` | Средний reward задания в кластере |
| 8 | `log_count` | `log(interaction_count + 1)` задания в кластере |
| 9 | `is_conceptual` | 1.0 если `task_type == "conceptual"` |
| 10 | `is_word_problem` | 1.0 если `task_type == "word_problem"` |
| 11 | `is_mixed` | 1.0 если `task_type == "mixed"` |
| 12 | `n_steps_norm` | `min(n_steps, 5) / 5.0` |

### 6.2 Планируемые расширения (Пакет 2+)

При переходе на Thompson Sampling вектор может быть расширен:

| Индекс | Поле | Формула |
|--------|------|---------|
| 13 | `confidence` | Confidence mastery текущей KC |
| 14 | `hint_dependence` | Из student model |
| 15 | `guessing_rate` | Из student model |

Размерность `CONTEXT_DIM` станет 16. Миграция: новые модели инициализируются с `A = I(16)`, старые модели `A(13)` padding нулями до 16.

---

## 7. Exploration Contract

### 7.1 Текущие exploration mechanisms

| Механизм | Вероятность | Описание |
|----------|------------|----------|
| Cluster exploration | 20% | Задание с `interaction_count < 3` в кластере |
| Epsilon-greedy | 5% | Полностью случайное задание |
| LinUCB UCB | 75% | `theta^T x + alpha * sqrt(x^T A^-1 x)` |

### 7.2 Контракт на exploration по режимам (Пакет 2)

| Режим | Cluster explore | Epsilon | Exploit strategy |
|-------|----------------|---------|-----------------|
| `build` | 20% | 5% | Standard UCB/TS |
| `consolidate` | 10% | 10% | Bias toward easier tasks |
| `test` | 0% | 0% | Pure exploit, diagnostic tasks preferred |
| `diagnostic` | 0% | 0% | CAT-style selection (max info gain) |

---

## 8. Plan Lifecycle Contract

### 8.1 Текущие владельцы (проблема D1)

| Решение | Владелец сейчас | Владелец целевой |
|---------|-----------------|-----------------|
| Создание плана | Macro | Macro |
| Advance step | Macro (`check_and_advance`) | Macro |
| Completion threshold | Profile (0.9) + Macro (configurable) | **Macro only** |
| Regression detection | Profile (`consecutive_errors >= 3`) | Macro |
| MicroSummary evaluation | Macro | Macro |
| Replan | Macro (STUB) | Macro (real replan) |

### 8.2 Целевой контракт (Пакет 2)

- Profile отвечает ТОЛЬКО за: student state, mastery update, interaction log.
- Macro отвечает за ВСЕ решения по плану: advance, complete, replan, remedial, alerts.
- Единый threshold: `plan.mastery_threshold` (default 0.80), только в Macro.
- Retrieval сообщает Macro если произошёл fallback с plan KC (через Kafka или return field).

---

## 9. Mastery Update Contract

### 9.1 Текущая формула (EMA smooth_update)

```
effective_lr = lr * streak_bonus * surprise_bonus
new_p = p_decayed + effective_lr * (score - p_decayed) + transit * (1 - new_p)
result = blend(p_decayed, new_p, hint_weight * role_weight)
```

Параметры:
- `SMOOTH_LR = 0.15` (базовый)
- `SMOOTH_TRANSIT = 0.02`
- `SURPRISE_K = 1.0`
- `HALF_LIFE_DAYS = 30.0`
- `PERFORMANCE_DECAY_THRESHOLD = 3` (consecutive errors)
- `PERFORMANCE_DECAY_FACTOR = 0.75`
- Streak bonus: `min(0.40, lr * (1 + 0.15 * consecutive_correct))` при `recent_accuracy >= 0.6`
- Hint weight: `0.4` если `hints_used > 0`, иначе `1.0`
- Role weight: `1.0` primary, `0.3` secondary

### 9.2 Контракт на BKT-параметры

| Параметр | Значение | Статус |
|----------|----------|--------|
| `p_transit` | 0.1 | Передаётся, НЕ используется в production EMA |
| `p_slip` | 0.1 | Передаётся, НЕ используется в production EMA |
| `p_guess` | 0.2 | Передаётся, НЕ используется в production EMA |

Эти параметры используются только в BKT-симуляторе (`bkt_environment.py`) для Q-learning.
Production mastery update полностью на EMA (`smooth_update`).

---

## 10. ZPD Contract

### 10.1 Фильтрация

```
KC in ZPD если:
  1. grade_introduced in [student_grade - 2, student_grade + 1]
  2. Все strong prereqs (strength >= 0.5) имеют mastery >= 0.7
  3. mastery_effective < 0.95
```

### 10.2 Целевой IRT pre-filter (Пакет 2)

```
KC + Task in candidate set если:
  P_correct(mastery, irt_difficulty) in [0.2, 0.9]
```

Задания с P_correct < 0.2 или > 0.9 исключаются ДО ранжирования бандитом.

---

## 11. Clustering Contract

### 11.1 Текущее

- Алгоритм: KMeans, `n_clusters=15` (hardcoded).
- Фичи: mastery vector.
- Хранение: `/tmp/learnity_centroids.npy` (теряется при перезагрузке).
- Переназначение: при `task_count == 15`.

### 11.2 Целевое (Пакет 1 + Пакет 3)

- **Пакет 1:** Перенести хранение в PostgreSQL.
- **Пакет 3:** KMeans -> GMM + BIC, soft membership.

---

## 12. Константы и Magic Numbers

Полный список констант, подлежащих выносу в конфиг (Пакет 1):

### Retrieval

| Константа | Значение | Файл |
|-----------|----------|------|
| `CLUSTER_EXPLORE_RATE` | 0.20 | `retrieval/main.py` |
| `CLUSTER_EXPLORE_THRESHOLD` | 3 | `retrieval/main.py` |
| `EPSILON_GREEDY_RATE` | 0.05 | `retrieval/main.py` |
| `KC_COOLDOWN_WINDOW` | 6 | `retrieval/main.py` |
| `KC_COOLDOWN_MAX` | 3 | `retrieval/main.py` |
| `PHASE1_TASK_THRESHOLD` | 15 | `retrieval/main.py` |
| `CONTEXT_DIM` | 13 | `retrieval/linucb.py` |
| `ALPHA` | 0.5 | `retrieval/linucb.py` |
| `TARGET_ZPD_ACCURACY` | 0.65 | `retrieval/selector.py` |
| `SUMMARY_WINDOW` | 20 | `retrieval/micro_summary.py` |

### Profile / BKT

| Константа | Значение | Файл |
|-----------|----------|------|
| `SMOOTH_LR` | 0.15 | `profile/bkt.py` |
| `SMOOTH_TRANSIT` | 0.02 | `profile/bkt.py` |
| `SURPRISE_K` | 1.0 | `profile/bkt.py` |
| `HALF_LIFE_DAYS` | 30.0 | `profile/bkt.py` |
| `PERFORMANCE_DECAY_THRESHOLD` | 3 | `profile/bkt.py` |
| `PERFORMANCE_DECAY_FACTOR` | 0.75 | `profile/bkt.py` |

### Graph / ZPD

| Константа | Значение | Файл |
|-----------|----------|------|
| `MASTERY_THRESHOLD` | 0.7 | `graph/zpd.py` |
| `MASTERY_CEILING` | 0.95 | `graph/zpd.py` |
| `STRONG_PREREQ` | 0.5 | `graph/zpd.py` |

### Gateway

| Константа | Значение | Файл |
|-----------|----------|------|
| `_BETA` | 0.3 | `gateway/main.py` |
| `_GAMMA` | 0.1 | `gateway/main.py` |

### Macro

| Константа | Значение | Файл |
|-----------|----------|------|
| `mastery_threshold` | 0.80 (default) | `macro/main.py` |

### Profile plan check

| Константа | Значение | Файл |
|-----------|----------|------|
| Completion threshold | 0.9 (hardcoded) | `profile/main.py:526` |
| Regression threshold | 3 (consecutive errors) | `profile/main.py:531` |

### Clustering

| Константа | Значение | Файл |
|-----------|----------|------|
| `N_CLUSTERS` | 15 | `clustering/cluster.py` |

---

## 13. Kafka Event Contract

| Event | Producer | Consumer | Payload |
|-------|----------|----------|---------|
| `MICRO_SUMMARY` | Retrieval | Macro | MicroSummary dict (section 5) |
| `STUDENT_REGISTERED` | Gateway | — | `{student_id, grade}` |
| `TASK_ANSWERED` | Gateway | — | `{student_id, task_id, score, ...}` |
| `TASK_REQUESTED` | Gateway | — | `{student_id}` |
| `MASTERY_REQUESTED` | Gateway | — | `{student_id}` |
| `PLAN_REQUESTED` | Gateway | — | `{student_id, mode, ...}` |

---

## 14. Метрики недоступные по решению

| Метрика | Причина |
|---------|---------|
| `time_spent` (реальное время на задание) | Метрика никогда не будет доступна. Не планировать фичи вокруг неё. Поле `time_spent_seconds` в `interactions` — заглушка. |
