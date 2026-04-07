# Архитектура системы Learnity: Macro + Micro + Интеграция

> Документ фиксирует итоговые архитектурные решения по двухуровневой системе рекомендаций.
> Основан на обсуждении 2026-03-31.

---

## Общая схема

```
Ученик задаёт цель (или работает свободно)
              ↓
      ┌───────────────┐
      │  MACRO LEVEL  │  Стратегия: какие KC изучать и в каком порядке
      └───────┬───────┘
              │ MacroDirective (KC, mode, budget)
              ↓
      ┌───────────────┐
      │  MICRO LEVEL  │  Тактика: какое конкретное задание выдать
      └───────┬───────┘
              │ MicroSummary (velocity, frustration, mastery delta)
              └──────────────────────────────────→ обратно в Macro
```

---

## 1. Макро-уровень: Стратегия

### 1.1 Два режима работы

**Режим 1 — Target Mastery**

Ученик выбирает тему и целевой уровень освоения. Система строит план — упорядоченный путь по пре-реквизитному подграфу.

```
Вход:  target_kc_id, target_mastery (например, 0.80), student_id
Выход: упорядоченный список KC-шагов с tasks_budget на каждый
```

Алгоритм: RL-политика на пре-реквизитном подграфе, обученная офлайн в BKT-симуляции. Отдельная политика на каждый кластер (15 штук). Инициализация из кластерного центроида — это и есть персонализация.

Функция награды:
```
reward = Δmastery(target_kc)
       + 0.1 × Σ Δmastery(prereq) × edge_strength(prereq → target)
       - 0.01  (штраф за шаг, ищем короткий путь)
```

**Режим 2 — Coverage**

Ученик задаёт горизонт (количество задач) и область графа. Система максимизирует покрытие.

```
Вход:  task_budget (N задач), scope_filter (grade/subject), coverage_variant
Выход: динамическая политика выбора KC из ZPD
```

Три варианта покрытия (выбирается при старте):
- `count` — максимизировать количество KC с mastery ≥ порога
- `mass` — максимизировать суммарный прирост mastery
- `frontier` — приоритет новым KC (никогда не практиковавшимся), бонус ×2

Алгоритм: RL с бюджетом как частью состояния.
```
State: mastery_by_subject×grade (сжатый) + tasks_remaining + velocity_by_subject
```
Персонализация: `learning_speed(kc, student) = α × cluster_prior + (1-α) × personal_history`
где `α = min(1.0, n_practiced_in_subject / 20)`.

### 1.2 Оценка tasks_to_mastery

```
tasks_to_mastery(kc, m_current, m_target, student) =
    α × cluster_observed_avg[cluster][subject][difficulty_bin]
    + (1 - α) × simulation_estimate(bkt_params, m_current, m_target)
```

Используется для: прогноза длины плана, распределения бюджета, обратной связи ученику.

### 1.3 Когда макро пересматривает план

**Плановые триггеры** (что наступит раньше):
- mastery сдвинулась на +0.1 от начала работы над KC
- потрачено 15 задач на KC

**Внеплановые триггеры:**
- `OnTargetAchieved` — mastery ≥ target_mastery
- `OnFrustration` — frustration_count ≥ 2 И velocity ≈ 0
- `OnBudgetAlert` — tasks_remaining < 20% от начального бюджета
- `OnClusterShift` — кластер ученика изменился

### 1.4 Три уровня пересмотра плана

| Уровень | Сигнал | Реакция |
|---------|--------|---------|
| 1 — лёгкий | Первая фрустрация | `difficulty_mode = consolidate` |
| 2 — средний | Фрустрация + mastery(prereq) < 0.70 | Вставить слабый prereq как срочный следующий шаг |
| 3 — тяжёлый | `velocity_ratio < 0.3` на 3+ шагах подряд | Полный перепланировщик от текущего mastery |

**OnClusterShift** всегда → уровень 3 (перестроить весь план с новым кластерным prior).

### 1.5 Компоненты для реализации

| Компонент | Описание |
|-----------|----------|
| `PrereqSubgraphExtractor` | BFS назад по Neo4j от target KC |
| `BKTEnvironment` | Симулятор для обучения RL офлайн |
| `TasksToMasteryEstimator` | Гибридная оценка (cluster + simulation) |
| `TargetMasteryPolicy` | RL per cluster, Режим 1 |
| `CoveragePolicy` | RL с бюджетом, Режим 2 |
| `PlanLifecycleManager` | Хранение плана, триггеры, эскалация |
| DB миграция | Добавить `goal_type`, `deadline`, `mastery_threshold`, `difficulty_mode`, `tasks_budget`, `tasks_spent`, `require_test` в `learning_plans` / `plan_steps` |

---

## 2. Микро-уровень: Тактика (Рекомендательная система)

### 2.1 Текущий пайплайн (основа)

```
ZPD кандидаты → выбор KC → задания из TaskBank → IRT фильтр → LinUCB → задание ученику
```

**Трёхуровневый выбор задания:**
- 20% — cluster exploration (задание с < 3 показов в кластере)
- 5%  — ε-greedy (случайное)
- 75% — LinUCB exploitation

### 2.2 Изменения в существующем коде

**1. IRT фильтр перед LinUCB (критично)**

До выбора задания отсекать те, где P(correct) вне целевого диапазона.

```python
θ = logit(mastery_kc)  # = log(mastery / (1 - mastery))
P_correct = 1 / (1 + exp(-(θ - task.irt_difficulty)))

# Фильтр по difficulty_mode:
# build:       keep if 0.60 ≤ P_correct ≤ 0.75
# consolidate: keep if 0.75 ≤ P_correct ≤ 0.90
# test:        keep if 0.45 ≤ P_correct ≤ 0.60
```

**2. Заполнить x[1..4] в контекстном векторе LinUCB**

```python
x[1..4] = [mastery(prereq_1), mastery(prereq_2), mastery(prereq_3), mastery(prereq_4)]
```

Дополнительный запрос к Graph за пре-реквизитами KC. Даёт LinUCB контекст о базе знаний ученика.

**3. seen_tasks с TTL**

Не фильтровать задания показанные более 30 дней назад. Если mastery KC упала из-за decay — задание снова доступно для повторения.

### 2.3 Новое

**4. difficulty_mode от макро → диапазон P(correct)**

| Режим | P(correct) | Когда |
|-------|-----------|-------|
| `build` | 60–75% | Стандарт, освоение новой KC |
| `consolidate` | 75–90% | После фрустрации, нужен успех |
| `test` | 45–60% | Перед переходом к следующему шагу |

В свободном режиме (без плана) — всегда `build`.

**5. Bridge bonus для заданий-мостиков**

```python
# После LinUCB scoring
if next_plan_kc in task.secondary_kcs:
    score += 0.1  # задание развивает текущую и прогревает следующую тему
```

**6. MicroSummary → макро**

Вычисляется и отправляется при каждом плановом или внеплановом триггере.

```python
MicroSummary:
  kc_id, mastery_before, mastery_after
  velocity          = Δmastery / tasks_spent
  velocity_ratio    = velocity / cluster_predicted_velocity
  frustration_count = кол-во раз consecutive_errors ≥ 3
  avg_score, hint_rate, irt_residual, tasks_spent
```

**7. Theta в профиле (перспектива)**

Хранить θ per KC как отдельное поле — для более точной IRT-оценки по мере накопления данных.

### 2.4 Свободный режим (без плана)

Поведение как сейчас: ZPD + subject rotation + LinUCB. Difficulty_mode = `build`. Нет директивы сверху.

---

## 3. Интеграция: как слои взаимодействуют

### 3.1 Поток данных

```
┌─────────────────────────────────────────────────────────┐
│                    MACRO PLANNER                        │
│                                                         │
│  Строит план → записывает MacroDirective в plan_steps   │
│  Слушает MicroSummary → пересматривает план при нужде   │
└──────────────────────┬──────────────────────────────────┘
                       │ MacroDirective
                       │ (current_kc, target_mastery,
                       │  difficulty_mode, tasks_budget,
                       │  require_test)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   MICRO SELECTOR                        │
│                                                         │
│  Читает MacroDirective из plan_steps                    │
│  IRT фильтр по difficulty_mode                          │
│  LinUCB + bridge bonus                                  │
│  Обновляет tasks_spent после каждой задачи              │
│                                                         │
│  При триггере → вычисляет MicroSummary → в Kafka        │
└──────────────────────┬──────────────────────────────────┘
                       │ MicroSummary
                       └────────────────────────→ Macro
```

### 3.2 Переход между шагами плана

```
Нормальный прогресс:
  mastery < target - 0.05  → difficulty_mode = build (обычная работа)
  mastery ≥ target - 0.05  → difficulty_mode = test (проверка прочности)
  test пройден             → OnTargetAchieved → следующий шаг
  test провален            → вернуться в build

Опционально (require_test = False):
  mastery ≥ target         → OnTargetAchieved сразу, без test-фазы
```

### 3.3 Обработка проблем

```
Нормальный поток:
  ответ ученика → BKT update → tasks_spent++ → (триггер?) → MicroSummary → Macro

При фрустрации (уровень 1):
  Macro меняет difficulty_mode = consolidate в plan_steps
  Micro читает на следующем запросе → выдаёт более лёгкие задания

При слабом пре-реквизите (уровень 2):
  Macro вставляет новый plan_step с prereq KC перед текущим
  status текущего шага = suspended
  Micro переключается на prereq KC

При системной проблеме (уровень 3):
  Macro запускает полный перепланировщик
  Новый план записывается в learning_plans / plan_steps
  Micro продолжает работу с новым активным шагом

При смене кластера (OnClusterShift):
  Аналогично уровню 3 — полный перепланировщик с новым кластерным prior
```

### 3.4 Транспорт событий

| Событие | Транспорт | Направление |
|---------|-----------|-------------|
| MacroDirective | PostgreSQL (plan_steps) | Macro → Micro |
| tasks_spent обновление | PostgreSQL (plan_steps) | Micro → DB |
| MicroSummary | Kafka topic `micro_summaries` | Micro → Macro |
| Plan rebuilt | Kafka topic `macro_directives` | Macro → Micro |

---

## 4. Что ещё не реализовано (отложено)

| Компонент | Причина отложить |
|-----------|-----------------|
| `CollaborativePlanner` | Нужны реальные данные ≥ 200 учеников с историей успеха |
| Theta per KC в профиле | Достаточно logit(mastery) на старте |
| Полный IRT с калибровкой (a, c параметры) | Нужны данные, пока a=1, c=0 |
| RL обучение онлайн | Начать с офлайн симуляции, перейти позже |

---

## 5. Порядок реализации

```
Фаза 1 — Микро-улучшения (не ломают существующее)
  ├── IRT фильтр в retrieval/main.py
  ├── Заполнить x[1..4] в _build_context()
  └── seen_tasks с TTL в Profile service

Фаза 2 — Интерфейс Macro ↔ Micro
  ├── DB миграция plan_steps (difficulty_mode, tasks_budget, tasks_spent, require_test)
  ├── Micro читает difficulty_mode → применяет к IRT фильтру
  ├── Micro обновляет tasks_spent
  ├── MicroSummary вычисление и отправка в Kafka
  └── Bridge bonus в LinUCB scoring

Фаза 3 — Macro Planner (ядро)
  ├── PrereqSubgraphExtractor
  ├── BKTEnvironment (симулятор)
  ├── TasksToMasteryEstimator
  ├── TargetMasteryPolicy (RL, Режим 1)
  ├── PlanLifecycleManager (триггеры + эскалация)
  └── DB миграция learning_plans (goal_type, mastery_threshold, require_test)

Фаза 4 — Coverage Policy (Режим 2)
  ├── CoveragePolicy (RL с бюджетом)
  └── LearningSpeedModel (cluster + personal)
```
