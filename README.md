# Learnity RecSys

**Адаптивная рекомендательная система для персонализированного обучения математике (K-12)**

[![CI](https://github.com/Kew1710/learnity-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/Kew1710/learnity-recsys/actions)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Tests](https://img.shields.io/badge/tests-256%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Проблема

Традиционное обучение математике — one-size-fits-all: все ученики получают одинаковые задания, независимо от уровня знаний. Это приводит к:

- **Фрустрации** — слишком сложные задания для слабых учеников
- **Скуке** — слишком лёгкие задания для сильных
- **Неэффективному обучению** — нет адаптации к индивидуальному темпу

## Решение

Learnity RecSys — ML-система, которая в реальном времени подбирает оптимальное задание для каждого ученика, опираясь на:

- **Текущий уровень знаний** (Bayesian Knowledge Tracing)
- **Зону ближайшего развития** (ZPD-фильтрация по графу пререквизитов)
- **Контекстный многорукий бандит** (Thompson Sampling для выбора задания)
- **Диагностику причин неуспеха** (prereq_gap / content_gap / uncertain_estimate / regression)

---

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Gateway    │────▶│   Profile    │
│  (клиент)    │◀────│  (FastAPI)   │     │  BKT/EMA     │
└─────────────┘     └──────┬───────┘     │  Confidence  │
                           │             │  Behavioral  │
                           │             └──────────────┘
                           │
                    ┌──────▼───────┐     ┌──────────────┐
                    │  Retrieval   │────▶│    Graph     │
                    │  Thompson TS │     │  KC prereqs  │
                    │  IRT filter  │     │  ZPD compute │
                    │  Diag. CAT   │     └──────────────┘
                    └──────┬───────┘
                           │             ┌──────────────┐
                    ┌──────▼───────┐     │  Task Bank   │
                    │    Macro     │     │  задания по  │
                    │  Plan lifecycle    │  KC и grade   │
                    │  Diagnostics │     └──────────────┘
                    │  Transitions │
                    └──────────────┘
                           │
              ┌────────────▼────────────┐
              │      Clustering         │
              │   GMM + BIC (auto k)    │
              └─────────────────────────┘

Инфраструктура: PostgreSQL · Neo4j · Apache Kafka · Docker
```

### Сервисы

| Сервис | Назначение | Ключевые алгоритмы |
|--------|-----------|-------------------|
| **Profile** | Состояние ученика, mastery, поведенческие сигналы | BKT smooth_update, confidence, guessing/hint detection |
| **Retrieval** | Выбор следующего задания | Thompson Sampling, IRT pre-filter, Diagnostic CAT |
| **Macro** | Управление учебным планом | Plan lifecycle, MicroSummary, Diagnostic reason layer |
| **Graph** | Граф знаний (KC), пререквизиты | ZPD computation, prerequisite traversal |
| **Clustering** | Кластеризация учеников | GMM + BIC (автоподбор k), soft membership |
| **Gateway** | Единая точка входа, оркестрация | Reward computation, Kafka event routing |
| **Task Bank** | Хранение и выдача заданий | Фильтрация по KC, grade, difficulty |

---

## ML Pipeline

### 1. Knowledge Tracing (оценка знаний)

```
Ответ ученика → smooth_update (EMA + streak bonus + surprise bonus)
              → apply_decay (забывание: half-life 30 дней)
              → confidence = f(attempts, stability, recency)
```

- **Mastery**: EMA-оценка `P(знает KC)` с адаптивным learning rate
- **Confidence**: уверенность в оценке — учитывает количество попыток, стабильность и давность
- **Behavioral signals**: guessing_rate (угадывание) и hint_dependence (зависимость от подсказок)

### 2. Task Selection (выбор задания)

```
ZPD кандидаты → IRT pre-filter → Thompson Sampling → задание
                  (mode-aware)     (Bayesian bandit)
```

- **ZPD фильтрация**: grade ± 2, prereq mastery > 0.7, KC mastery < 0.95
- **IRT pre-filter**: убирает задания с P(correct) вне [floor, ceiling] для текущего режима
- **Thompson Sampling**: сэмплирует θ ~ N(A⁻¹b, v²A⁻¹), выбирает задание с max θᵀx
- **Режимы обучения**: build / consolidate / test / diagnostic — разные exploration rates и IRT boundaries

### 3. Cold Start (диагностический CAT)

```
Новый ученик → 5-8 адаптивных заданий → калибровка mastery priors
               (максимум информации Фишера)
```

- Выбирает KC с минимумом наблюдений, задания где P(correct) ≈ 0.5
- Быстро калибрует mastery вместо чисто grade-based priors

### 4. Macro Policy (управление планом)

```
MicroSummary → Diagnostic Layer → Plan Actions
(velocity, frustration, avg_score)   (причина проблемы)   (consolidate, insert_prereq, replan)
```

- **Diagnostic layer**: различает 4 причины неуспеха:
  - `prereq_gap` — слабый пререквизит
  - `content_gap` — мало заданий нужной сложности
  - `uncertain_estimate` — недостаточно данных
  - `regression` — забывание
- **Transitions logging**: записывает (state, action, reward, next_state) для будущего offline RL

### 5. Clustering (сегментация учеников)

- **GMM + BIC**: автоматический подбор числа кластеров (вместо KMeans k=15)
- **Soft membership**: вероятностная принадлежность к кластерам
- **Transfer learning**: новый ученик получает Thompson Sampling prior из кластера

---

## Результаты и метрики

| Метрика | Значение | Описание |
|---------|----------|----------|
| Тесты | 256 passed | Unit-тесты на все ML-компоненты |
| Сервисы | 7 микросервисов | Полная production-ready архитектура |
| ML-модели | 5 алгоритмов | BKT, Thompson TS, IRT, GMM+BIC, Diagnostic CAT |
| Decision log | 6 полей | Полная трассировка каждого решения |
| Transitions | JSONB | Offline RL-ready логирование |

### Offline Evaluation Pipeline

```bash
python -m tools.offline_eval           # полная оценка
python -m tools.offline_eval --kt-only # только Knowledge Tracing
```

Метрики KT: Brier Score, Log-Loss, AUC, Calibration по бакетам.
Метрики Macro: распределение действий, reward per action, diagnosis accuracy.

---

## Быстрый старт

### Интерактивная демонстрация (без БД)

```bash
pip install streamlit numpy scikit-learn plotly
streamlit run demo/app.py
```

5 интерактивных вкладок: симуляция обучения, mastery & confidence, Thompson vs LinUCB, кластеризация, архитектура.

### Полный запуск (Docker)

```bash
# Поднять инфраструктуру
docker-compose up -d

# Применить миграции
alembic upgrade head

# Запустить сервисы
uvicorn services.gateway.main:app --port 8000
uvicorn services.profile.main:app --port 8001
uvicorn services.graph.main:app --port 8002
uvicorn services.task_bank.main:app --port 8003
uvicorn services.retrieval.main:app --port 8004
uvicorn services.macro.main:app --port 8005
```

### Тесты

```bash
pip install pytest pytest-asyncio httpx respx numpy scikit-learn sqlalchemy aiokafka fastapi
python -m pytest services/ tools/tests/ -q
```

---

## Структура проекта

```
learnity-recsys/
├── services/
│   ├── profile/          # Mastery tracking (BKT, confidence, behavioral signals)
│   ├── retrieval/         # Task selection (Thompson Sampling, IRT, CAT)
│   ├── macro/             # Plan lifecycle (diagnostics, transitions)
│   ├── graph/             # KC graph, ZPD computation
│   ├── clustering/        # Student clustering (GMM + BIC)
│   ├── gateway/           # API gateway, Kafka routing
│   └── task_bank/         # Task storage and retrieval
├── shared/
│   ├── config.py          # Centralized ML config (env var overrides)
│   ├── db.py              # Database connection
│   └── schemas.py         # Shared data schemas
├── migrations/            # Alembic migrations (18 versions)
├── demo/
│   └── app.py             # Streamlit interactive demo
├── tools/
│   ├── offline_eval.py    # Offline evaluation pipeline
│   ├── simulation.py      # Student simulation
│   └── ab_eval.py         # A/B test evaluation
├── docs/
│   ├── ml_contract.md     # ML system contract (14 sections)
│   └── audit_summary.md   # Architecture audit & roadmap
├── docker-compose.yml
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Технологический стек

| Категория | Технологии |
|-----------|-----------|
| **ML/DS** | NumPy, scikit-learn, Thompson Sampling, BKT, IRT, GMM |
| **Backend** | Python, FastAPI, SQLAlchemy, Pydantic |
| **Data** | PostgreSQL, Neo4j, Apache Kafka |
| **Infra** | Docker, Docker Compose, GitHub Actions CI |
| **Visualization** | Streamlit, Plotly |

---

## Roadmap

- [x] **Пакет 1**: Confidence, behavioral signals, decision log, centralized config
- [x] **Пакет 2**: IRT pre-filter, learning modes, plan ownership, diagnostics, transitions
- [x] **Пакет 3**: Thompson Sampling, Diagnostic CAT, GMM+BIC, offline eval
- [ ] **Пакет 4**: DKT dual-track, offline macro policy, neural bandit evaluation

---

## Автор

**Александр Григорьев** — [alex.grig04.2@gmail.com](mailto:alex.grig04.2@gmail.com)
