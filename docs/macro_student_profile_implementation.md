# MacroStudentProfile: План Внедрения

## Цель документа

Этот документ нужен как рабочее ТЗ для кодингового агента.

Задача: внедрить в проект новый слой `MacroStudentProfile`, набор небольших оценщиков для макро-планирования и интеграцию этого слоя с текущим `SubgraphQAgent`, не ломая существующий macro pipeline.

Документ описывает:

- что именно нужно построить;
- в какие модули это встраивать;
- в каком порядке вносить изменения;
- какие интерфейсы должны появиться;
- какие тесты и инварианты обязательны.


## 1. Проблема

Сейчас макро-план строится в основном по:

- `mastery`;
- `graph of prerequisites`;
- эвристикам `tasks_estimator`;
- `SubgraphQAgent` для `target_mastery`.

Этого недостаточно для хорошего cold start и ранней персонализации.

Два ученика могут иметь похожий `mastery`-вектор, но сильно отличаться по:

- скорости продвижения;
- склонности застревать;
- требуемому бюджету задач;
- чувствительности к пробелам в пререквизитах;
- устойчивости после ошибок.

Эти свойства сейчас не собраны в отдельный production-объект и не подаются в `SubgraphQAgent` как часть state.


## 2. Целевое решение

Нужно ввести слой `MacroStudentProfile`.

Итоговая схема:

1. После диагностического блока и далее по мере обучения система строит и обновляет `MacroStudentProfile`.
2. Из `MacroStudentProfile` и локального контекста темы считаются несколько макро-оценок:
   - `predicted_tasks_to_mastery`
   - `predicted_stall_risk`
   - `predicted_regression_risk`
3. `SubgraphQAgent` продолжает строить маршрут по подграфу, но получает более богатое состояние.
4. Budget, pacing и prereq strictness задаются не агентом напрямую, а profile/estimators слоем.

Ключевой принцип:

- маленькие оценщики не заменяют `SubgraphQAgent`;
- они улучшают state и параметры планировщика;
- сам выбор порядка KC остаётся в существующем macro planner.


## 3. Что не делаем в этой итерации

Не делать:

- новый RL-агент с нуля;
- отдельный macro-clustering слой;
- heavy ML-модели или нейросети;
- subject-driven логику как основную ось профиля;
- полный переписанный planner.

Первая реализация должна быть:

- graph-native;
- rule-based внутри оценщиков;
- совместимой с последующей заменой rule-based логики на learned models.


## 4. Где это живёт в текущей архитектуре

Основные точки интеграции:

- `services/macro/main.py`
- `services/macro/policy_mode1.py`
- `services/macro/tasks_estimator.py`
- `services/macro/plan_lifecycle.py`
- `services/macro/clients.py`
- `services/macro/kafka_consumer.py`

Новые модули:

- `services/macro/student_profile.py`
- `services/macro/profile_builder.py`
- `services/macro/estimators.py`
- `services/macro/tests/test_student_profile.py`
- `services/macro/tests/test_estimators.py`


## 5. Целевой объект MacroStudentProfile

`MacroStudentProfile` должен быть production-объектом, который можно:

- построить после диагностики;
- обновлять после шагов плана;
- читать при генерации нового плана;
- логировать для offline-анализа.

Пока не использовать широкие метки вроде "алгебра/геометрия" как главную ось.
Опора должна быть на граф и локальный контекст цели.

### 5.1. Минимальный состав полей

```python
class MacroStudentProfile(BaseModel):
    student_id: UUID
    version: int
    updated_at: datetime

    uncertainty_level: float
    mastery_confidence_mean: float
    weak_prereq_fraction: float
    target_subgraph_mastery_mean: float

    learning_speed_global: float
    learning_speed_recent: float
    tasks_to_gain_01_mastery: float
    recovery_after_error: float

    frustration_risk: float
    stall_risk_baseline: float
    regression_risk_baseline: float

    pacing_mode: str
    budget_multiplier: float
    prereq_strictness: float
    test_readiness_bias: float
    step_granularity: float
```

### 5.2. Смысл полей

- `uncertainty_level`: насколько профиль ещё сырой; обычно доля KC с низкой confidence.
- `mastery_confidence_mean`: средняя уверенность в mastery по relevant KC.
- `weak_prereq_fraction`: доля слабых пререквизитов внутри подграфа цели.
- `target_subgraph_mastery_mean`: средний mastery по целевому подграфу.
- `learning_speed_global`: долгосрочный темп роста mastery.
- `learning_speed_recent`: недавний темп роста mastery.
- `tasks_to_gain_01_mastery`: сколько задач обычно нужно, чтобы поднять mastery на `0.1`.
- `recovery_after_error`: как быстро ученик восстанавливается после ошибок.
- `frustration_risk`: склонность уходить в серию слабых попыток без прогресса.
- `stall_risk_baseline`: общий baseline-риск застревания.
- `regression_risk_baseline`: baseline-риск распада mastery после успеха.
- `pacing_mode`: `"careful" | "balanced" | "aggressive"`.
- `budget_multiplier`: множитель к budget шага.
- `prereq_strictness`: насколько строго уходить в remedial/prereq path.
- `test_readiness_bias`: насколько рано переводить шаг в `test`.
- `step_granularity`: насколько крупными делать шаги плана.


## 6. Источники данных для профиля

Первая версия должна использовать уже существующие источники:

- `profile mastery`
- `profile confidence`
- результаты диагностического блока / CAT
- `bandit_log.raw_score`
- `bandit_log.hints_used`
- `bandit_log.time_spent_seconds`
- `bandit_log.mastery_delta`
- `bandit_log.irt_difficulty`
- актуальный подграф цели
- статус прошлых `plan_steps`

Не требовать новых внешних сервисов.


## 7. Graph-native признаки

Не строить профиль вокруг coarse `subject`.

Использовать признаки, привязанные к графу:

- средний mastery по подграфу цели;
- средний confidence по подграфу цели;
- доля слабых direct prerequisites;
- доля слабых bottleneck prerequisites;
- глубина цели в графе;
- средняя глубина слабых узлов;
- прогресс в локальной ветке графа;
- скорость продвижения в текущем подграфе;
- доля шагов, которые завершились через `consolidate` или `insert_prereq`.


## 8. Оценщики

Нужны три небольших production-оценщика.

### 8.1. `estimate_tasks_to_mastery`

Назначение:

- оценить, сколько задач нужно на конкретную KC.

Интерфейс:

```python
def estimate_tasks_to_mastery(
    profile: MacroStudentProfile,
    kc_id: str,
    current_mastery: float,
    target_mastery: float,
    weak_prereq_fraction: float,
    confidence: float | None = None,
) -> float:
    ...
```

Первая реализация:

- rule-based;
- использует текущий `tasks_estimator` как базу;
- затем модифицирует оценку через `budget_multiplier`, uncertainty и prereq penalties.

### 8.2. `estimate_stall_risk`

Назначение:

- оценить вероятность того, что ученик застрянет на этой KC/ветке.

Интерфейс:

```python
def estimate_stall_risk(
    profile: MacroStudentProfile,
    kc_id: str,
    weak_prereq_fraction: float,
    confidence: float | None = None,
    graph_depth: int | None = None,
) -> float:
    ...
```

Первая реализация:

- rule-based score `0..1`;
- использует `frustration_risk`, `stall_risk_baseline`, weak prereqs, low confidence, depth penalty.

### 8.3. `estimate_regression_risk`

Назначение:

- оценить вероятность того, что после прохождения темы mastery быстро распадётся.

Интерфейс:

```python
def estimate_regression_risk(
    profile: MacroStudentProfile,
    kc_id: str,
    confidence: float | None = None,
) -> float:
    ...
```

Первая реализация:

- rule-based;
- использует `regression_risk_baseline`, confidence и прошлые регрессивные паттерны, если доступны.


## 9. ProfileBuilder

Нужен отдельный builder, который создаёт и обновляет `MacroStudentProfile`.

Новый модуль:

- `services/macro/profile_builder.py`

Минимальные функции:

```python
async def build_initial_macro_profile(
    student_id: UUID,
    target_kc_id: str | None = None,
) -> MacroStudentProfile:
    ...

async def refresh_macro_profile(
    student_id: UUID,
    target_kc_id: str | None = None,
) -> MacroStudentProfile:
    ...
```

### 9.1. Когда строить initial profile

После достаточной диагностики.

Базовое правило:

- если у ученика меньше `15` решённых задач, строим только черновой профиль;
- если у ученика `15-30` диагностических/ранних задач, строим первый рабочий `MacroStudentProfile`;
- после этого профиль обновляется регулярно.

### 9.2. Когда обновлять profile

Обязательные триггеры:

- после завершения `plan_step`;
- после `replan`;
- после сильного `frustration`/`plateau`;
- после завершения диагностического блока;
- опционально после cluster shift на микро-уровне.


## 10. Хранилище

Нужна отдельная таблица `macro_student_profiles`.

Минимальная схема:

```sql
CREATE TABLE macro_student_profiles (
    student_id UUID PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    uncertainty_level DOUBLE PRECISION NOT NULL,
    mastery_confidence_mean DOUBLE PRECISION NOT NULL,
    weak_prereq_fraction DOUBLE PRECISION NOT NULL,
    target_subgraph_mastery_mean DOUBLE PRECISION NOT NULL,

    learning_speed_global DOUBLE PRECISION NOT NULL,
    learning_speed_recent DOUBLE PRECISION NOT NULL,
    tasks_to_gain_01_mastery DOUBLE PRECISION NOT NULL,
    recovery_after_error DOUBLE PRECISION NOT NULL,

    frustration_risk DOUBLE PRECISION NOT NULL,
    stall_risk_baseline DOUBLE PRECISION NOT NULL,
    regression_risk_baseline DOUBLE PRECISION NOT NULL,

    pacing_mode TEXT NOT NULL,
    budget_multiplier DOUBLE PRECISION NOT NULL,
    prereq_strictness DOUBLE PRECISION NOT NULL,
    test_readiness_bias DOUBLE PRECISION NOT NULL,
    step_granularity DOUBLE PRECISION NOT NULL
);
```

Также нужна история для offline-анализа.

Варианты:

- либо `macro_student_profile_snapshots`;
- либо логирование snapshot в отдельную таблицу при каждом обновлении.

Для первой итерации достаточно:

- production table `macro_student_profiles`;
- snapshot table `macro_student_profile_snapshots`.


## 11. Интеграция с текущим macro planner

### 11.1. Что меняется в `services/macro/main.py`

Перед построением плана нужно:

1. загрузить или пересчитать `MacroStudentProfile`;
2. прокинуть его в `_build_mode1_plan`;
3. использовать его при расчёте budgets и state features;
4. сохранить профиль, если он был пересчитан.

Новая логика:

- план нельзя строить только по `mastery`;
- `MacroStudentProfile` должен быть обязательной частью pipeline для `target_mastery`.

### 11.2. Что меняется в `services/macro/tasks_estimator.py`

Текущий estimator должен стать базовым слоем.

Нужно:

- оставить текущую функцию как `base estimate`;
- поверх неё ввести profile-aware adjustment;
- не ломать существующие вызовы сразу, а добавить новую обёртку.

Рекомендуемый интерфейс:

```python
def estimate_with_profile(...):
    ...
```

### 11.3. Что меняется в `services/macro/policy_mode1.py`

`SubgraphQAgent` пока не переписывать полностью.

Нужно сделать `v2` состояния:

- сохранить текущую Q-learning механику;
- расширить state encoder;
- не переводить агент сразу на непрерывный state.

Минимальный путь:

1. добавить profile-aware state key;
2. дискретизировать не только mastery, но и некоторые profile-derived признаки;
3. использовать их при training и inference одинаково.


## 12. State для SubgraphQAgent v2

Текущее состояние агента основано почти только на `mastery`.

Нужно перейти к состоянию:

```python
state = {
    "mastery_by_node": ...,
    "mean_confidence_bin": ...,
    "weak_prereq_fraction_bin": ...,
    "learning_speed_recent_bin": ...,
    "stall_risk_baseline_bin": ...,
    "pacing_mode": ...,
}
```

### 12.1. Что именно подавать в state v2

Обязательные признаки:

- mastery bins по узлам подграфа;
- средняя confidence по relevant KC;
- weak prereq fraction;
- recent learning speed bin;
- baseline stall risk bin;
- pacing mode.

Не включать в state v2 слишком много признаков.
Иначе табличный Q-agent взорвётся по размеру.

### 12.2. Как кодировать

- `mastery`: текущая дискретизация остаётся;
- `confidence`: 3-5 бинов;
- `weak_prereq_fraction`: 3-5 бинов;
- `learning_speed_recent`: 3-5 бинов;
- `stall_risk_baseline`: 3-5 бинов;
- `pacing_mode`: 3 категории.

Если размер state-space начнёт резко расти, сначала убрать наименее полезный признак, а не увеличивать сложность агента.


## 13. Что оставляем вне SubgraphQAgent

Агент должен выбирать маршрут по KC.

Не нужно перекладывать на него напрямую:

- `tasks_budget`;
- `difficulty_mode`;
- `require_test`;
- решение о `consolidate`;
- точную remedial-политику.

Это должен делать profile/plan layer.

Итог:

- агент выбирает порядок KC;
- profile/estimators слой задаёт параметры шага.


## 14. Как profile влияет на план

`MacroStudentProfile` должен влиять на четыре вещи.

### 14.1. Порядок KC

Через enriched state для `SubgraphQAgent`.

### 14.2. Budget на шаг

`tasks_budget` должен зависеть от:

- базовой оценки стоимости темы;
- `budget_multiplier`;
- weak prereq penalty;
- uncertainty penalty.

### 14.3. Строгость работы с prereqs

`prereq_strictness` влияет на:

- насколько рано вставлять remedial/prereq step;
- насколько охотно делать обход по базе вместо прямого движения к цели.

### 14.4. Test timing

`test_readiness_bias` влияет на:

- когда шаг можно переводить в `test`;
- насколько высокий confidence нужен до тестовой фазы.


## 15. Порядок внедрения

Работу разбить на 5 PR.

### PR1. Каркас MacroStudentProfile

Сделать:

- миграцию `macro_student_profiles`;
- snapshot table;
- `services/macro/student_profile.py`;
- `services/macro/profile_builder.py`;
- pydantic/dataclass model;
- загрузку/сохранение профиля;
- базовые unit-тесты.

Критерий готовности:

- профиль можно построить и сохранить без интеграции с planner.

### PR2. Rule-based builder и estimators

Сделать:

- вычисление initial profile из текущих данных;
- `estimate_tasks_to_mastery`;
- `estimate_stall_risk`;
- `estimate_regression_risk`;
- unit-тесты на формулы и граничные случаи.

Критерий готовности:

- по фиксированным входным данным profile и estimators дают детерминированный результат.

### PR3. Интеграция в macro planning

Сделать:

- загрузку profile в `services/macro/main.py`;
- profile-aware budget calculation;
- интеграцию в `_build_mode1_plan`;
- расширение state encoder в `policy_mode1.py`;
- тесты на построение плана с разными profile settings.

Критерий готовности:

- два ученика с одинаковым mastery, но разным profile, получают отличающийся macro plan или step parameters.

### PR4. Обновление profile по ходу обучения

Сделать:

- refresh profile после завершения шага;
- refresh profile после важных lifecycle events;
- snapshot logging;
- integration tests на обновление profile по истории шагов.

Критерий готовности:

- profile меняется после накопления нового поведения, а не остаётся статичным.

### PR5. Dataset hooks для будущих learned models

Сделать:

- логирование labels для `tasks_to_mastery`, `stall`, `regression`;
- `tools/stats.py` или отдельный tool для выгрузки baseline метрик;
- docs по offline training contract.

Критерий готовности:

- данные для будущих обучаемых оценщиков копятся автоматически.


## 16. Тест-план

### 16.1. Unit tests

Нужны тесты на:

- build initial profile;
- refresh profile;
- estimator formulas;
- state discretization;
- state key stability;
- migration/repository roundtrip.

### 16.2. Integration tests

Нужны тесты на:

- `create_new_plan` использует сохранённый profile;
- отсутствие profile приводит к его построению, а не к падению;
- profile влияет на `tasks_budget`;
- profile влияет на state агента;
- profile refresh происходит после шага плана.

### 16.3. Behavioral tests

Нужны тесты на сценарии:

1. Высокая uncertainty + высокий frustration.
Ожидаем:
- `pacing_mode=careful`
- повышенный `tasks_budget`
- более строгий `prereq_strictness`

2. Высокая скорость + хорошее восстановление после ошибки.
Ожидаем:
- `pacing_mode=aggressive`
- пониженный `tasks_budget`
- ранний `test readiness`

3. Слабые пререквизиты в целевом подграфе.
Ожидаем:
- выше `weak_prereq_fraction`
- более консервативный маршрут
- выше риск застревания


## 17. Инварианты

Обязательные инварианты:

- planner работает, даже если profile отсутствует, потому что умеет его построить;
- profile builder детерминирован на одинаковом входе;
- profile нельзя silently игнорировать в `target_mastery` path;
- state encoding одинаков для training и inference;
- расширение state не должно ломать старые policy cache keys без явного versioning.


## 18. Версионирование policy

После добавления нового state старые policy-файлы нельзя считать совместимыми.

Нужно:

- изменить `policy_path(...)`, добавив version marker;
- либо ввести `policy_state_version`;
- либо включить version в имя файла policy.

Пример:

```python
def policy_path(cluster_id: int, target_kc: str, state_version: str = "v2") -> str:
    return f"models/policy_{state_version}_cluster_{cluster_id}_{target_kc}.pkl"
```


## 19. Риски

### 19.1. Слишком большой state-space агента

Риск:

- табличный Q-agent станет плохо обучаться или раздуется.

Митигировать:

- ограничить число новых признаков;
- использовать coarse bins;
- добавить state versioning;
- сначала внедрять только 3-4 profile-derived признака.

### 19.2. Профиль будет формально существовать, но реально не влиять на план

Риск:

- внедрим таблицы и builder, но planner останется прежним.

Митигировать:

- обязательные integration tests, где профиль меняет budget/state/route.

### 19.3. Слишком сложные rule-based формулы

Риск:

- код станет трудно сопровождать;
- дальнейшая замена на ML усложнится.

Митигировать:

- держать оценщики простыми;
- строить их через небольшие независимые функции;
- не смешивать profile building и planner logic.


## 20. Что будет во второй итерации

После стабилизации production-пайплайна можно заменить внутренности оценщиков на обучаемые модели.

Интерфейсы должны остаться теми же:

- `estimate_tasks_to_mastery`
- `estimate_stall_risk`
- `estimate_regression_risk`

На этой фазе добавляются:

- labels из истории plan steps;
- offline evaluation;
- простые regression/classification models.

Важно:

- интерфейсы не менять;
- planner не должен знать, rule-based estimator внутри или learned.


## 21. Definition of Done

Подход считается внедрённым, когда выполнено всё ниже:

1. В БД есть production table и snapshot table для `MacroStudentProfile`.
2. Для нового ученика после диагностического блока строится initial profile.
3. При создании `target_mastery` плана macro planner использует profile.
4. `SubgraphQAgent` работает на state v2, а не только на `mastery`.
5. `tasks_budget` и prereq behavior зависят от profile.
6. После завершения шага profile обновляется.
7. Есть unit и integration tests, подтверждающие влияние profile на результат.
8. Policy artifacts версионированы и не конфликтуют со старым state.


## 22. Минимальный чеклист для кодингового агента

1. Добавить миграции для `macro_student_profiles` и snapshots.
2. Создать `student_profile.py`, `profile_builder.py`, `estimators.py`.
3. Реализовать build/load/save/refresh profile.
4. Добавить rule-based estimator functions.
5. Прокинуть profile в `services/macro/main.py`.
6. Расширить state encoder в `services/macro/policy_mode1.py`.
7. Сделать profile-aware budget calculation.
8. Добавить versioning policy artifacts.
9. Написать unit и integration tests.
10. Обновить docs при необходимости.

