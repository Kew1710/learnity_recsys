# Learnity — Обзор системы

> Актуально на: 2026-04-06

---

## 1. Цель проекта

**Learnity** — адаптивная рекомендательная система заданий по математике для школьников.

Три задачи:

1. **Подбор заданий** — каждое задание оптимально под текущий уровень ученика: не слишком легко (скучно), не слишком сложно (фрустрация). Целевой диапазон P(верный ответ) = 65–75% (зона ближайшего развития).

2. **Учебный план** — система строит долгосрочный план: для достижения целевой темы последовательно закрывает пробелы в пре-реквизитах.

3. **Измерение прогресса** — BKT в реальном времени оценивает знание каждой KC, отслеживает забывание, собирает метрики обучения.

---

## 2. Архитектура

### 2.1 Микросервисы

Система состоит из независимых сервисов. Агенты не используются — только детерминированные алгоритмы и обученные модели.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Gateway :8005                           │
│  POST /students   POST /students/{id}/answer   GET /next-task   │
└────────────┬──────────────────┬────────────────────┬────────────┘
             │                  │                    │
     ┌───────▼──────┐  ┌────────▼──────┐  ┌─────────▼────────┐
     │  Profile     │  │   Retrieval   │  │   Macro Planner  │
     │  :8001       │  │   :8004       │  │   :8006          │
     │              │  │               │  │                  │
     │  mastery     │  │  ZPD + LinUCB │  │  plan lifecycle  │
     │  BKT update  │  │  task select  │  │  Q-learning      │
     └──────────────┘  └───────┬───────┘  └──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌───────▼──────┐  ┌─────▼──────────┐
     │  Graph        │ │  Task Bank   │  │  Clustering    │
     │  :8002        │ │  :8003       │  │  (in-process)  │
     │               │ │              │  │                │
     │  Neo4j DAG    │ │  PostgreSQL  │  │  k-means       │
     │  KC + edges   │ │  tasks+parts │  │  student types │
     └───────────────┘ └──────────────┘  └────────────────┘

Инфраструктура (Docker): PostgreSQL · Neo4j · Kafka
```

### 2.2 Горячий путь (синхронный, <500мс)

Всё что происходит пока ученик ждёт следующего задания:

```
Ученик отправил ответ
  → Gateway: POST /students/{id}/answer
    → Profile: обновить mastery (BKT)
    → Retrieval: POST /recommend
        → Profile: GET mastery + seen_tasks
        → Graph: GET ZPD-кандидаты
        → TaskBank: GET tasks/by_kc (с exclude фильтром)
        → LinUCB: выбрать задание
        → bandit_log: записать контекстный вектор
  ← вернуть {mastery_update, next_task}
```

### 2.3 Холодный путь (асинхронный, Kafka)

После ответа — в фоне:

```
Событие answer_submitted →
  → Macro Planner: MicroSummary → evaluate_plan → action
  → Аналитика: сохранить interaction
  → IRT калибровка: обновить параметры задания (отложено)
```

**Почему Kafka:** история событий сохраняется неделями → можно переиграть, переобучить модель, восстановить профиль с нуля.

---

## 3. Принятые решения

### 3.1 Алгоритм оценки знаний — BKT + EMA

**Решение:** модифицированный BKT на основе EMA (Exponential Moving Average).

Классический BKT работает с бинарными ответами и четырьмя параметрами (P(L0), P(T), P(S), P(G)). Мы упростили до EMA-формулы с расширениями:

```
new_mastery = p + effective_lr × (score − p)
```

Расширения поверх базового BKT:
- **Непрерывный score** [0.0–1.0] вместо бинарного
- **Decay:** `mastery_effective = mastery_stored × 0.5^(days / half_life_days)` — знания забываются
- **Surprise boost:** если ученик решил сложнее чем ожидалось, `lr` увеличивается
- **Consecutive correct streak:** серия правильных ответов ускоряет калибровку
- **Подсказки снижают вес:** `update_weight = 0.4 if hints_used else 1.0`
- **secondary_kcs:** косвенный сигнал обновляет KC с весом 0.3
- **Performance decay:** 3+ ошибки подряд → `mastery × 0.75` (пересмотр уверенности)
- **review_mode:** флаг на студенте, отключает decay при активном освоении

**Почему не DKT/DKVMN:** требуют тысячи студентов для обучения. BKT работает с одним.

### 3.2 Выбор задания — LinUCB (контекстный бандит)

**Проблема:** как выбрать конкретное задание из пула, зная KC и mastery студента?

**Решение:** LinUCB — линейный контекстный бандит (Upper Confidence Bound).

```
score(задание) = θᵀx + α√(xᵀA⁻¹x)
                 ────────  ──────────────
                exploitation  exploration
```

Одна модель (матрица A 13×13 + вектор b) на каждую KC, хранится в PostgreSQL как BYTEA. Обновляется онлайн после каждого ответа:
```
A ← A + xxᵀ
b ← b + reward × x
```

**Контекстный вектор (13 измерений):**
| Индекс | Фича |
|--------|------|
| x[0] | mastery текущей KC |
| x[1..4] | mastery пре-реквизитов (по убыванию силы) |
| x[5] | consecutive_errors |
| x[6] | grade / 11.0 |
| x[7] | avg_reward задания в кластере |
| x[8] | log(interaction_count + 1) |
| x[9] | is_conceptual (0/1) |
| x[10] | is_word_problem (0/1) |
| x[11] | is_mixed (0/1) |
| x[12] | n_steps / 5.0 |

**Стратегия выбора (в порядке приоритета):**
1. **Phase 1 (первые 15 заданий):** эвристика на основе ZPD, без бандита — данных ещё нет
2. **Cluster exploration 20%:** задание которое кластер изучил <3 раз — кооперативный exploration
3. **ε-greedy 5%:** полностью случайное задание
4. **LinUCB 75%:** exploitation с exploration bonus

### 3.3 Фичи заданий — task_type + n_steps

**Решение:** добавлены два новых атрибута на уровне `Part`:

| Поле | Значения | Смысл |
|------|----------|-------|
| `task_type` | `procedural / conceptual / word_problem / mixed` | когнитивный тип задания |
| `n_steps` | 1–5 | количество шагов до ответа |

**Распределение в seed:** 40% procedural, 30% conceptual, 20% word_problem, 10% mixed.

Разные типы дают разный learning effect: conceptual задания труднее, но при правильном ответе дают больший прирост знаний. LinUCB учится: для какого профиля студента и уровня mastery оптимален каждый тип.

### 3.4 Кластеризация студентов

**Проблема:** LinUCB со свежей моделью (A=I) — чистый exploration для каждого нового студента.

**Решение:** k-means кластеризация студентов по mastery-профилю. Новый студент получает `cluster_id`, наследует LinUCB-модель своего кластера — сразу получает опыт похожих учеников.

```
Новый студент → assign_cluster → cluster_id → load cluster LinUCB model
```

Студенческая LinUCB-модель и кластерная — разные: индивидуальная накапливается поверх кластерной.

### 3.5 Zone of Proximal Development (ZPD)

**Решение:** граф KC фильтруется в два этапа:

1. **Grade filter:** KC с `grade_introduced > student.grade + 1` не показываются
2. **ZPD логика:**
   - KC с mastery < порога И все сильные пре-реквизиты освоены → `ready` (основная работа)
   - KC с mastery ≈ пороговому → `review` (повторение)
   - Освоенные KC → `mastered` (не показываем)

Слабые пре-реквизиты (strength < 0.5) не блокируют переход — допускается частичное знание базы.

### 3.6 Cold Start

При создании студента (`POST /students?grade=8`) инициализируются mastery через хранимую процедуру:

```
grade_introduced < grade - 2   → mastery = 0.95   (давно освоено)
grade_introduced = grade - 2   → mastery = 0.90
grade_introduced = grade - 1   → mastery = 0.75
grade_introduced = grade        → mastery = 0.50   (текущий класс)
grade_introduced > grade        → mastery = 0.0    (ещё не проходили)
```

В симуляции `true_mastery` инициализируется из cold_start: если `vis > 0.50` → `true = vis`, иначе → `true = profile.initial_true_mastery`.

### 3.7 Макро-планировщик — два режима

**Режим 1: target_mastery** (основной сценарий)

Цель: достичь mastery целевой KC. Алгоритм:
1. BFS назад по графу пре-реквизитов от target KC
2. Включить KC где `mastery < threshold − 0.05` (не освоена и не почти)
3. Q-learning агент (`SubgraphQAgent`) обучается на BKT-симуляторе и выдаёт оптимальную последовательность KC
4. Приоритет: освоить пре-реквизиты в топологическом порядке

**Режим 2: coverage** (покрытие программы)

Цель: максимальное покрытие KC за ограниченный бюджет заданий. Три варианта стратегии: `count` (по кол-ву KC), `mass` (по трудозатратам), `frontier` (KC на границе освоенного).

### 3.8 Plan Lifecycle — реакция на прогресс

Macro Planner получает `MicroSummary` от Retrieval через Kafka и принимает решения:

| Сигнал | Условие | Действие |
|--------|---------|----------|
| `advance` | `mastery[current_step] >= threshold` | перейти к следующему шагу |
| `frustration` | avg_score < 0.35 за 10 заданий | вернуться к пре-реквизиту |
| `plateau` | mastery_velocity ≈ 0 за 15 заданий | сменить стратегию/сложность |
| `false_mastery` | mastery высокая, затем accuracy упала | не продвигать, дать подтверждающие задания |

### 3.9 Модель симуляции студента

Симуляция (для тестирования без реальных учеников) строится на двух векторах:

**`true_mastery`** — реальные знания студента, скрыты от системы:
```python
# Логистический рост от верных ответов:
delta = learning_rate × GROWTH_SCALE × score × (1 − true_mastery)
new_true = min(1.0, true_mastery + delta)
```
Знания растут монотонно. learning_rate — индивидуальный параметр профиля (fast_learner=0.28, slow_learner=0.04).

**`visible_mastery`** — оценка системы (BKT), обновляется через API:
```
new_vis = vis + SMOOTH_LR × (score − vis)    # EMA, SMOOTH_LR=0.15
```

MAE(true, visible) — основная метрика качества системы: насколько быстро и точно она узнаёт реального студента.

**Профили студентов:**

| Профиль | learning_rate | initial_true | Особенность |
|---------|--------------|--------------|-------------|
| fast_learner | 0.28 | 0.05 | Быстро усваивает, мало ошибается |
| slow_learner | 0.04 | 0.05 | Медленно, часто ошибается, сильный penalty на conceptual |
| average | 0.12 | 0.08 | Типичный студент |
| advanced | 0.15 | 0.70 | Уже знает материал, система должна быстро откалиброваться |

Каждый профиль имеет `type_difficulty_mod` (насколько нелюбимый тип задания субъективно сложнее) и `type_learning_mod` (прирост знаний при верном ответе по типу).

---

## 4. Реализация архитектурных идей в коде

### 4.1 BKT — `services/profile/bkt.py`

```python
# Decay перед обновлением
mastery_effective = mastery_stored × 0.5^(days / half_life_days)

# Smooth update с расширениями
def smooth_update(p_mastery, score, lr, consecutive_correct, irt_difficulty):
    surprise = score - expected_p(p_mastery, irt_difficulty)
    effective_lr = lr × streak_factor(consecutive_correct) × (1 + SURPRISE_K × max(0, surprise))
    return p + effective_lr × (score - p) + transit × (1 - p)
```

`bkt.py` — чистые функции без side effects.  
Статус верификации на 2026-04-06: `pytest services/profile/tests/test_bkt.py -q` → `36 passed`.

### 4.2 ZPD — `services/graph/zpd.py`

```python
# KC попадает в ZPD если:
# 1. mastery ниже порога освоения
# 2. все "сильные" пре-реквизиты (strength >= 0.5) уже освоены
def is_ready(kc, mastery, prereqs) -> bool:
    if mastery.get(kc) >= MASTERY_CEILING: return False  # уже освоено
    strong_prereqs = [p for p in prereqs if p.strength >= 0.5]
    return all(mastery.get(p.kc_id, 0) >= PREREQ_THRESHOLD for p in strong_prereqs)
```

### 4.3 LinUCB — `services/retrieval/linucb.py` + `main.py`

Модель персистится как пара BYTEA (матрица A + вектор b) в таблице `bandit_model`:

```python
# Загрузка: сначала пытаемся персональную, потом кластерную, потом init
model = load_student_model(student_id, kc_id) 
     or load_cluster_model(cluster_id, kc_id)
     or LinUCBModel.init(kc_id, cluster_id)

# Выбор: UCB-score для каждого кандидата
for task in candidates:
    x = _build_context(mastery, grade, avg_reward, count, task_type, n_steps)
    s = model.score(x)
    s -= 0.5 × abs(p_correct - TARGET_ZPD_ACCURACY)  # ZPD accuracy penalty
    if next_plan_kc in task.secondary_kcs: s += 0.1   # bridge bonus

# Обновление (отложенное, после получения reward через PATCH)
model.update(x, reward)
save_student_model(model)
save_cluster_model(model)  # знания студента обогащают кластер
```

### 4.4 Граф пре-реквизитов — `services/macro/prereq_extractor.py`

```python
def extract_prereq_subgraph(target_kc_id, mastery, graph, threshold):
    cutoff = threshold - MASTERY_GAP  # KC "почти освоена" если mastery >= threshold - 0.05
    queue = deque([target_kc_id])
    
    while queue:
        kc = queue.popleft()
        if mastery.get(kc, 0) < cutoff or kc == target_kc_id:
            nodes.append(kc)          # KC требует работы
        for prereq in graph[kc]:
            if prereq.mastery < cutoff:
                queue.append(prereq)  # BFS назад только по не-освоенным
```

Результат — подграф из KC требующих работы. Для `kc_irrational_eq` с threshold=0.85 у студента класса 8: 5 KC grade-8 + 5 KC grade-7 = 10 шагов плана.

### 4.5 Q-learning для плана — `services/macro/policy_mode1.py`

```python
# Состояние: дискретизированный mastery-профиль подграфа (5 бинов на KC)
def _state_key(mastery, node_order) -> tuple:
    return tuple(int(mastery.get(kc, 0.0) * 5) for kc in node_order)

# Агент обучается на BKTEnvironment за N эпизодов
# Действие: выбрать KC для следующего задания
# Reward: прирост mastery target KC
# γ = 0.9, ε-greedy exploration
```

Q-таблица сохраняется в `models/` как pickle, загружается при старте сервиса.

### 4.6 Симуляция — `tools/simulation.py`

Симуляция заменяет реальных учеников для тестирования pipeline:

```python
# true_mastery — скрытые знания студента (логистический рост)
delta = lr × GROWTH_SCALE × score × (1 − true_m)
new_true = min(1.0, true_m + delta)

# Ответ генерируется из true_mastery через IRT
effective_diff = irt_difficulty + type_difficulty_mod[task_type]
score = IRT(true_mastery, effective_diff, p_slip, p_guess)

# visible_mastery обновляется через реальный API (BKT в БД)
resp = POST /students/{id}/answer(score)
visible_mastery[kc] = resp.mastery_update
```

Цикл `run_single()` вызывает реальные API: Profile, Retrieval, TaskBank — симуляция тестирует весь горячий путь целиком.

### 4.7 Схема данных — `services/task_bank/models.py`, `services/profile/models.py`

```
tasks → parts             (один task — несколько частей)
  parts.primary_kcs       ARRAY(String)  — основные KC задания
  parts.secondary_kcs     ARRAY(String)  — косвенные KC
  parts.irt_difficulty    Float          — параметр IRT
  parts.task_type         String         — procedural|conceptual|word_problem|mixed
  parts.n_steps           Integer        — шагов до ответа

students → mastery        (один студент — много KC)
  mastery.probability     Float          — BKT-оценка знания KC
  mastery.last_practiced  DateTime       — для расчёта decay
  mastery.consecutive_correct Integer   — для streak bonus в BKT

learning_plans → plan_steps  (один план — последовательность шагов)
  plan_steps.kc_id        String         — KC шага
  plan_steps.status       String         — pending|active|completed
  plan_steps.priority     Float          — порядок в плане

bandit_model              — LinUCB A,b матрицы (BYTEA) per (kc_id, cluster_id)
bandit_log                — context_vector + reward per interaction
```

### 4.8 Изоляция сервисов и circular imports

`simulation.py` вызывает `sandbox.py` через отложенный импорт внутри функций:

```python
def run_single(...):
    from tools.sandbox import api_create_student, api_next_task  # lazy import
```

`sandbox.py` вызывает `simulation.py` аналогично:

```python
if cmd == "sim":
    from tools.simulation import sim_wizard   # lazy import
    sim_wizard(client)
```

Это разрывает circular import на уровне модулей.

---

## 5. Текущий статус реализации

| Компонент | Статус |
|-----------|--------|
| Gateway | ✅ Работает |
| Profile + BKT | ✅ Работает, streak-bug исправлен, `test_bkt` зелёный (36/36) |
| Graph + ZPD | ✅ Работает, 34 теста |
| TaskBank | ✅ Работает, seed 142 KC × 100 заданий |
| Retrieval + LinUCB | ✅ Работает |
| Clustering | ✅ Работает (in-process) |
| Macro Planner | ✅ Работает (режим 1 + режим 2) |
| Симуляция | ✅ Режимы 1–9, логистический рост true_mastery |
| Фичи task_type/n_steps | ✅ В схеме, seed, LinUCB контексте |
| Kafka (холодный путь) | ⚠️ Частично: macro consumer выполняет auto-advance и plateau alerts |
| IRT калибровка | ⏸ Отложено |
| Авто-eval при плато | ⚠️ Частично: alert реализован, автопереплан пока не включён |
| Наблюдаемость hot path | ✅ Добавлены структурные логи в Gateway/Retrieval (ошибки fallback + latency/source) |

---

## 6. Открытые архитектурные вопросы

1. **Plan lifecycle в фоне всё ещё узкий** — consumer делает auto-advance и plateau-alert, но не применяет `insert_prereq/replan` автоматически.

2. **IRT floor** — при низкой mastery IRT подбирает задания по уровню → P(correct) ≈ 65% → EMA equilibrium может не дойти до 0.80.

3. **Prereqs для KC grade=9** — часть KC 9-го класса остаётся без рёбер, из-за чего план может вырождаться в 1 шаг.

4. **Производительность LinUCB** — `A^-1` пересчитывается на каждый candidate-score; при росте пула заданий нужен инкрементальный апдейт обратной матрицы.
