# Learnity: Единый План Развития Системы

## Цель документа

Это не список разрозненных проблем, а единый пошаговый план развития Learnity.

Порядок приоритетов:

1. Сначала определяем **ML-видение системы** и целевые SOTA-направления.
2. Затем закрываем **архитектурные дыры и незамкнутые контуры**, чтобы ML-слой жил в устойчивой системе.
3. После этого оптимизируем **dev/infra и скорость разработки**.

Ключевой принцип: не внедрять более сильные модели в незамкнутую систему. Сначала фиксируем правильный state, сигналы и контуры обучения, затем усиливаем модели.

---

## 1. Целевое ML-видение

### 1.1 Что должно стать ядром системы

Learnity должна эволюционировать к трёхслойной ML-системе:

1. **Knowledge model**
   Оценивает не только `mastery`, но и `confidence`, forgetting, cold-start uncertainty и диагностические сигналы.

2. **Micro policy**
   Выбирает конкретное задание внутри текущей KC и режима обучения.
   Целевой путь: `LinUCB -> Thompson Sampling`.

3. **Macro policy**
   Выбирает следующую KC/шаг плана на основе реальных логов и диагностических сигналов.
   Целевой путь: от rule-based/Q-learning к offline policy learning на собранных transitions.

### 1.2 Что считать SOTA-прагматичным для Learnity

Не всё SOTA нужно внедрять сразу.

Приоритетный стек для этой системы:

1. **Knowledge tracing**
   Целевое направление: `simpleKT / AKT / SAINT+`, но не сейчас.
   Ближайший практический шаг: обогатить текущий mastery-слой uncertainty и sequence-aware features, не ломая весь pipeline.

2. **Bandit layer**
   Целевое направление: `Thompson Sampling`, а не neural bandit.
   Причина:
   - лучше согласуется с маленькими и шумными образовательными данными;
   - даёт естественную uncertainty-driven exploration;
   - проще объясним и дешевле в эксплуатации;
   - лучше подходит для промежуточного этапа до возможного neural policy.

3. **Cold start**
   Целевой путь: `rule-based prior + короткий diagnostic CAT`.
   Не CF-first, потому что CF потребует объёма и чистоты данных, которых пока нет.

4. **Macro learning**
   Целевой путь: сначала логирование transitions и offline evaluation, потом offline RL / imitation / ranking-политики.

5. **Student clustering**
   Целевой путь: `GMM + BIC` вместо жёсткого `KMeans(k=15)`.

### 1.3 Что нельзя делать раньше времени

Не надо раньше срока:

- заменять EMA сразу на трансформерный KT;
- переходить на neural contextual bandit;
- строить offline RL без хороших логов;
- усиливать политику, если reward и state ещё суррогатны.

---

## 2. Целевая логика системы

### 2.1 Целевое представление ученика

Система должна принимать решения не по одному `mastery`, а по следующему состоянию:

- `mastery_mean`
- `mastery_confidence`
- `attempts_count`
- `recent_accuracy`
- `forgetting_signal`
- `guessing_signal`
- `hint_dependence`
- `cluster_membership`
- `goal_mode`
- `plan_context`

Минимально допустимая промежуточная версия:

- оставить текущий `probability`;
- добавить `confidence`;
- начать обновлять `guessing_rate` и `hint_dependence`;
- логировать sequence features для будущего KT.

### 2.2 Целевое представление задания

Task должен стать first-class объектом рекомендаций, а не пассивной записью в банке.

Минимальный набор сигналов:

- `irt_difficulty`
- `task_type`
- `n_steps`
- `coverage_quality`
- `success_rate`
- `diagnostic_value`
- `content_availability_by_kc_and_band`

### 2.3 Целевой контур принятия решения

Целевая decision loop:

1. Оценить состояние ученика.
2. Определить режим: `diagnostic / build / consolidate / test`.
3. Определить KC или шаг плана.
4. Проверить контентную достижимость этого шага.
5. Выбрать task через bandit/policy.
6. Получить outcome.
7. Обновить student state.
8. Выполнить диагностику причины результата.
9. Обновить macro policy / plan / alerts.
10. Записать полный decision log для offline replay.

---

## 3. Исполнимый Пошаговый План

Ниже порядок выполнения. Его нужно идти сверху вниз, без перескоков через фазы.

---

## Фаза 0. Зафиксировать ML-контракт системы

### Цель

Перед любыми изменениями согласовать, что именно система оптимизирует и какие сигналы считает истинными.

### Что делаем

1. Зафиксировать основной online objective:
   `maximize learning progress under acceptable frustration`.

2. Зафиксировать вторичные objectives:
   - не зацикливаться на KC;
   - не выдавать недостижимые задания;
   - не путать контентный дефицит с проблемой ученика;
   - сохранять объяснимость решений.

3. Зафиксировать контракт student state:
   - какие поля обязательны сейчас;
   - какие поля добавляем в ближайшие 2 фазы;
   - какие поля будут только логироваться для будущих моделей.

4. Зафиксировать контракт reward:
   - текущий reward остаётся временным surrogate;
   - вводится отдельное поле `observed_outcome_features`;
   - future reward строится не только на `delta_mastery`.

### Результат фазы

Единое описание:

- student state;
- task state;
- reward contract;
- decision log contract.

---

## Фаза 1. Укрепить текущий ML-контур без смены моделей

### Цель

Не менять пока базовую математику радикально, а сделать текущий контур более правдоподобным и пригодным для будущего SOTA.

### Шаг 1. Добавить uncertainty в mastery

Что внедряем:

- `confidence = f(attempts_count, recency, stability)` как минимум;
- в перспективе можно перейти к `alpha/beta`, но на этом шаге не обязательно.

Зачем:

- система перестанет одинаково трактовать `0.5 после 2 попыток` и `0.5 после 50`;
- появится осмысленный input для ZPD и переходов по плану.

### Шаг 2. Ввести динамический `p_guess` и behavioral signals

Что внедряем:

- обновление `guessing_rate`;
- обновление `hint_dependence`;
- использование их в mastery update и decision logging.

Зачем:

- система начнёт различать “угадывает” и “знает”.

### Шаг 3. Сделать learning modes реальными режимами

Сейчас `difficulty_mode` слишком слабый.

Нужно сделать:

- `build`: обычное обучение;
- `consolidate`: ниже сложность, выше вероятность успеха;
- `test`: ниже exploration, выше диагностическая ценность;
- `diagnostic`: короткие задачи на быструю калибровку.

Меняем не только target difficulty, но и:

- exploration policy;
- candidate filtering;
- success criteria;
- logging reason.

### Шаг 4. Ввести настоящий IRT pre-filter

Не штраф после scoring, а фильтр кандидатов до выбора.

Зачем:

- bandit не должен ранжировать откровенно неподходящие задачи.

### Шаг 5. Сделать MicroSummary доступным и без активного плана

Что внедряем:

- вычисление MicroSummary не только для `plan_steps`, но и для free mode;
- отдельный lightweight lifecycle для учеников без плана.

### Результат фазы

Текущий EMA/BKT-слой остаётся, но становится:

- более устойчивым;
- менее слепым;
- лучше пригодным как источник фич для следующего шага.

---

## Фаза 2. Перестроить micro policy: LinUCB -> Thompson Sampling

### Цель

Заменить текущий бандит на policy, которая лучше работает с uncertainty и образовательным шумом.

### Почему именно Thompson Sampling

Для Learnity это более правильный следующий шаг, чем neural bandit:

- естественно кодирует uncertainty;
- проще для online-update;
- легче интерпретировать;
- меньше инфраструктурный риск;
- лучше подходит при умеренном объёме данных.

### Шаг 1. Подготовить policy interface

Сначала вынести policy API в абстракцию:

- `score_candidates(...)`
- `sample_action(...)`
- `update_policy(...)`

Нельзя делать замену алгоритма прямо поверх текущих веток `if phase1/control/treatment`.

### Шаг 2. Внедрить Thompson Sampling для task selection

Целевой вариант:

- per-KC policy;
- контекстный или semi-contextual TS;
- priors инициализируются кластером;
- student policy продолжает дообучаться поверх cluster prior.

### Шаг 3. Пересобрать exploration вокруг TS

После перехода нужно убрать избыточные костыли, которые TS уже покрывает:

- пересмотреть `epsilon-greedy`;
- пересмотреть cluster exploration;
- сохранить только то, что реально нужно как content exploration.

### Шаг 4. Сохранить phase-1 heuristic как отдельный режим

Cold start первых задач не надо ломать.

Нужно:

- оставить phase-1;
- добавить короткий diagnostic path;
- после phase-1 передавать ученика в TS policy.

### Результат фазы

Micro layer становится:

- более Bayesian;
- лучше работающим с uncertainty;
- более совместимым с будущим richer student state.

---

## Фаза 3. Усилить cold start и student modeling

### Цель

Убрать зависимость только от grade-based prior.

### Шаг 1. Добавить короткий diagnostic CAT

Для новых учеников:

- 5-10 калибровочных заданий;
- быстрый срез по ключевым prereq и текущему классу;
- на выходе обновлённый prior по mastery/confidence.

### Шаг 2. Заменить жёсткий KMeans на GMM + BIC

Что даёт:

- автоподбор структуры кластеров;
- soft membership;
- более корректный cluster prior для policy.

### Шаг 3. Сделать перекластеризацию полноценным событием

При cluster shift:

- переинициализировать student TS policy из нового cluster prior;
- инициировать пересмотр плана;
- логировать причину cluster shift.

### Результат фазы

Student model становится менее грубой и лучше ведёт себя на старте.

---

## Фаза 4. Замкнуть архитектуру вокруг единого владельца плана

### Цель

После усиления ML-состояния убрать логическую расщеплённость архитектуры.

### Шаг 1. Сделать Macro единственным owner-ом плана

Нужно:

- убрать lifecycle-решения по плану из `Profile`;
- оставить в `Profile` только student state и interactions;
- все advance/replan/remedial decisions сосредоточить в `Macro`.

### Шаг 2. Развести роли сервисов

- `Profile`:
  student state, mastery, behavior signals, interaction history.
- `Retrieval`:
  micro policy и task selection.
- `Macro`:
  lifecycle плана, причины эскалации, remedial actions, replan.
- `TaskBank`:
  не только выдача задач, но и контентные метрики.

### Шаг 3. Убрать молчаливый fallback с plan KC

Если для активного шага нет контента:

- это не обычный fallback;
- это architecture-level signal;
- создаётся alert и diagnostic reason.

### Результат фазы

Система перестаёт иметь два source of truth для плана.

---

## Фаза 5. Построить слой причинной интерпретации

### Цель

Перестать реагировать на симптомы без понимания причины.

### Целевой diagnostic layer

Модуль должен различать минимум 4 причины:

1. `prereq_gap`
2. `content_gap`
3. `uncertain_estimate`
4. `forgetting_or_regression`

Дополнительно можно выделять:

5. `task_quality_issue`
6. `guessing_behavior`

### Что модуль использует

- mastery + confidence;
- prereq profile;
- content availability by difficulty band;
- recent outcomes;
- mode of learning;
- cluster prior vs actual trajectory.

### Что модуль возвращает

- diagnosed_reason;
- recommended_action;
- confidence of diagnosis.

### Результат фазы

Macro перестаёт быть просто набором порогов и становится управляющим слоем с причинной логикой.

---

## Фаза 6. Сделать контент first-class constraint

### Цель

Чтобы система отличала “ученик не может” от “система не может подобрать”.

### Шаг 1. Ввести контентные агрегаты

По каждому `KC x difficulty_band`:

- task_count;
- active_task_count;
- success_rate;
- exploration_coverage;
- diagnostic coverage.

### Шаг 2. Использовать эти сигналы в retrieval и macro

- retrieval не должен молча идти в тупик;
- macro должен видеть, что шаг плана контентно недостижим;
- alerts должны различать learner issue и content issue.

### Шаг 3. Добавить task quality loop

Минимально:

- success rate by task;
- frustration contribution by task;
- suspected bad tasks.

### Результат фазы

Контент перестаёт быть скрытым ограничением и становится частью decision loop.

---

## Фаза 7. Подготовить систему к следующему поколению knowledge tracing

### Цель

Не внедрять transformer KT сейчас, но подготовить всё, чтобы переход стал возможен без архитектурного слома.

### Что делаем

1. Логируем sequence-ready события:
   - student_id
   - timestamp
   - kc_id
   - task_id
   - mode
   - correctness/score
   - hints
   - state snapshot before
   - state snapshot after

2. Вводим offline evaluation pipeline для mastery prediction.

3. Делаем benchmark:
   - current EMA/BKT baseline
   - richer-feature baseline
   - позже `simpleKT/AKT`.

### Результат фазы

Появляется безопасная дорожка к sequence-based KT без немедленной замены production-ядра.

---

## Фаза 7b. Интеграция DKT в production pipeline

### Цель

Заменить EMA на sequence-based knowledge tracing (simpleKT/AKT) в production через dual-track подход.

### Предусловия

- Фаза 7 завершена: sequence-ready логи собираются, offline benchmark показал превосходство DKT над EMA.
- Достаточный объём данных для обучения (минимум ~10k interactions).

### Шаг 1. Dual-track: EMA online + DKT offline

- DKT модель обучается offline на накопленных sequence логах.
- При каждом запросе mastery система возвращает два значения:
  - `mastery_ema` (текущий production)
  - `mastery_dkt` (предсказание DKT модели)
- Все решения (ZPD, plan, retrieval) продолжают использовать `mastery_ema`.
- `mastery_dkt` логируется для сравнения и калибровки.

### Шаг 2. Shadow mode: DKT принимает решения, но не исполняет

- Retrieval параллельно вычисляет рекомендацию по `mastery_dkt`.
- Логируется: "DKT выбрал бы задание X, EMA выбрал Y".
- Offline анализ: в каких случаях DKT даёт лучшие рекомендации.

### Шаг 3. A/B test: DKT vs EMA

- Часть учеников переводится на `mastery_dkt` как primary.
- Метрики: learning progress, frustration rate, plan completion rate.
- При положительных результатах — DKT становится primary.

### Шаг 4. Полный переход

- `mastery_dkt` становится единственным source of truth.
- EMA остаётся как fallback для новых учеников с < 5 interactions (DKT нужна минимальная последовательность).
- Периодическое переобучение DKT модели на свежих данных.

### Результат фазы

Knowledge tracing эволюционирует от точечной EMA-оценки к sequence-aware модели без рискованного big-bang перехода.

---

## Фаза 8. Подготовить macro learning к offline policy learning

### Цель

Уйти от tabular Q-learning на симуляторе к обучению на реальных логах, но через нормальную промежуточную инфраструктуру.

### Шаг 1. Логировать transitions

Нужна таблица/лог:

- `state`
- `action`
- `reward`
- `next_state`
- `done`
- `reason`

### Шаг 2. Ввести offline replay и оценку политик

До любой “умной” offline RL модели должны быть:

- dataset versioning;
- replay tooling;
- counterfactual/offline metrics;
- baseline policies.

### Шаг 3. Только потом пробовать offline policy models

В порядке:

1. heuristic ranking baselines
2. imitation / supervised policy
3. conservative offline RL

### Результат фазы

Macro SOTA появляется не как “магическая RL-модель”, а как следующий шаг после правильно собранных данных.

---

## Фаза 9. Усилить self-observability и experiment loop

### Цель

Сделать систему объяснимой самой себе и удобной для ML-разработки.

### Что внедряем

1. Расширенный decision log:
   - why this KC
   - why this task
   - exploration type
   - active plan step
   - candidate counts
   - diagnosed reason
   - fallback reason

2. Experiment tracking:
   - policy version
   - config version
   - feature set version
   - reward version

3. Metrics:
   - recommendation latency
   - content coverage failures
   - fallback-from-plan rate
   - advancement rate
   - frustration rate
   - reward drift

### Результат фазы

Появляется нормальный цикл:

`change -> measure -> compare -> rollback/ship`

---

## Фаза 10. Оптимизировать dev/infra только после логической стабилизации

### Цель

После того как ML-контуры и ownership слоёв стали правильными, упростить эксплуатацию и ускорить разработку.

### Шаг 1. Ускорить hot path

- переиспользовать HTTP clients;
- connection pooling;
- параллелизовать независимые вызовы;
- позже решить, нужно ли сливать `Profile` и `Retrieval`.

### Шаг 2. Вынести магические константы в единый конфиг

Нужно централизовать:

- thresholds;
- exploration params;
- mode ranges;
- cluster settings;
- alert settings.

### Шаг 3. Починить хранение кластеров

- убрать `/tmp` как источник истины;
- хранить центроиды в PostgreSQL или object storage;
- сделать scheduled refresh.

### Шаг 4. Нарастить observability

- Prometheus;
- OpenTelemetry;
- correlation IDs;
- dashboards по decision loop.

### Шаг 5. Снизить хрупкость схемы

Не обязательно полностью унифицировать ORM/raw SQL сразу, но нужно:

- добавить schema contract tests;
- зафиксировать migration safety checks;
- минимизировать “скрытые” SQL-зависимости.

---

## 4. Что делать прямо сейчас: ближайший исполнимый бэклог

Это первый реальный tranche работ. Его можно выполнять уже сейчас.

### Пакет 1. Немедленно

1. Добавить `confidence` в student state.
2. Добавить MicroSummary для учеников без плана.
3. Убрать молчаливый fallback с plan KC.
4. Добавить content-gap alert.
5. Ввести decision log fields в `bandit_log`.
6. Начать обновлять `guessing_rate` и `hint_dependence`.
7. Вынести magic numbers в конфиг.
8. Убрать хранение кластеров из `/tmp`.

### Пакет 2. Следом

1. Сделать настоящий IRT pre-filter.
2. Сделать learning modes отдельными policy regimes.
3. Перенести ownership плана полностью в `Macro`.
4. Ввести diagnostics reason layer v1:
   `prereq_gap / content_gap / uncertain_estimate / regression`.
5. Завести transitions logging для macro.

### Пакет 3. После стабилизации

1. Внедрить `Thompson Sampling` вместо `LinUCB`.
2. Добавить diagnostic CAT для cold start.
3. Перейти на `GMM + BIC` для кластеров.
4. Собрать offline evaluation pipeline для KT и macro policy.

### Пакет 4. После накопления данных

1. Benchmark sequence-based KT against current mastery pipeline.
2. Dual-track DKT: offline обучение + shadow mode + A/B test → полный переход (Фаза 7b).
3. Пробовать offline macro policy learning.
4. Решать, нужен ли neural bandit layer.

---

## 5. Главный принцип принятия решений

Для Learnity сейчас правильный порядок такой:

1. **Не усиливать модель, пока state и reward суррогатны.**
2. **Не оптимизировать инфраструктуру, пока ownership логики размазан.**
3. **Не внедрять сложный SOTA, пока нельзя replay-ить и объяснять решения.**
4. **Сначала Bayesian/pragmatic ML, потом heavy neural ML.**

Итоговый целевой путь:

`EMA/BKT + LinUCB + rule heuristics`

→ `EMA/BKT + confidence + Thompson Sampling + diagnostic layer`

→ `sequence-ready logs + offline evaluation + stronger macro policy`

→ `DKT dual-track → полная замена EMA при подтверждённом превосходстве`
