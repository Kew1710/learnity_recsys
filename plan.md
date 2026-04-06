# План реализации Learnity

> Статус: обсуждение
> Принцип: каждый этап — рабочий и тестируемый срез системы

---

## Этап 0 — Фундамент ✅

Инфраструктура и общий код, от которого зависят все сервисы.

- [x] `docker-compose.yml` — PostgreSQL, Neo4j, Kafka (KRaft mode)
- [x] `shared/schemas.py` — Pydantic-схемы: `Task`, `Part`, `IRT`, `Student`, `MasteryRecord`, `Interaction`
- [x] `shared/kafka_client.py` — продюсер и консьюмер (aiokafka)
- [x] `shared/db.py` — SQLAlchemy async engine + session factory
- [x] SQL-миграции (Alembic): таблицы `students`, `mastery`, `interactions`, `tasks`, `parts`
- [x] `shared/neo4j_client.py` — обёртка над neo4j async driver
- [x] `Makefile`: команды `up`, `down`, `migrate`, `test`, `lint`
- [x] `requirements-shared.txt` — зависимости

---

## Этап 1 — Сервис профиля (`services/profile`) ✅

Ядро системы — хранит и обновляет знания ученика.

- [x] `GET /students/{id}` — получить профиль
- [x] `POST /students` — создать ученика (с `grade`)
- [x] `GET /students/{id}/mastery` — получить все mastery-записи
- [x] `GET /students/{id}/mastery/{kc_id}` — mastery по конкретному KC
- [x] BKT update: `bkt.py` — `update_mastery()`
  - [x] Decay перед обновлением: `mastery_effective = mastery_stored * 0.5^(days/30)`
  - [x] Модифицированный BKT для непрерывного score `[0.0–1.0]`
  - [x] Учёт `hints_used` (вес обновления 0.4 vs 1.0)
  - [x] Учёт `primary_kcs` (вес 1.0) и `secondary_kcs` (вес 0.3)
- [x] `POST /students/{id}/interactions` — записать ответ + обновить mastery
- [x] Юнит-тесты BKT логики (21 тест, все зелёные)
- [x] Интеграционные тесты с реальным PostgreSQL (16 тестов, все зелёные)

---

## Этап 2 — Сервис графа знаний (`services/graph`) ✅

Хранит KC-ноды и рёбра в Neo4j.

- [x] Seed-скрипт: 24 KC по теме «Теорема Пифагора» (grade 4–8)
- [x] `GET /nodes/{kc_id}` → `get_node(kc_id)`
- [x] `GET /nodes/{kc_id}/prerequisites` → `get_prerequisites(kc_id)`
- [x] `POST /zpd` (body: `{mastery, student_grade}`) → ZPD-кандидаты
  - [x] Логика ZPD в `zpd.py` (чистые функции)
  - [x] Фильтр по `grade`, ceiling освоенности, слабые рёбра не блокируют
  - [x] Сортировка: ready → по difficulty
- [x] `GET /path?from={kc}&to={kc}` → `get_path(from_kc, to_kc)`
- [x] Юнит-тесты ZPD (19 тестов)
- [x] Интеграционные тесты с реальным Neo4j (15 тестов)

---

## Этап 3 — Сервис банка заданий (`services/task_bank`)

Хранит задания, отдаёт их по KC и параметрам IRT.

- [ ] Seed-скрипт: 5–10 заданий на каждый KC из seed-графа
  - [ ] Разные `answer_type`: numeric, multiple_choice
  - [ ] IRT-параметры для каждого part (вручную подобранные)
- [ ] `GET /tasks/{task_id}` — получить задание
- [ ] `GET /tasks?kc_id={id}&grade_min={n}` — задания по KC
- [ ] `POST /tasks` — добавить задание (для будущей генерации)
- [ ] Тесты

---

## Этап 4 — Сервис подбора тем (`services/retrieval`)

Определяет какие KC предложить ученику — ZPD + grade filter.

- [ ] `POST /recommend-topics` (body: `student_id`) → список KC-кандидатов
  - [ ] Запрос mastery ученика из сервиса профиля
  - [ ] Запрос ZPD-кандидатов из сервиса графа
  - [ ] Применение grade-фильтра
- [ ] Тесты (mock сервисов профиля и графа)

---

## Этап 5 — Сервис выбора задания (`services/ranking`)

По списку KC-кандидатов выбирает конкретное задание через IRT.

- [ ] IRT P(correct): `P = c + (1-c) / (1 + exp(-a*(theta - b)))`
- [ ] Целевой диапазон P(correct): 0.55–0.75 (оптимальная сложность)
- [ ] `POST /recommend-task` (body: `student_id`, `kc_candidates`) → `task_id`
  - [ ] Запрос заданий для KC-кандидатов из task_bank
  - [ ] Вычислить `theta` ученика из mastery
  - [ ] Выбрать задание ближайшее к целевому P(correct)
  - [ ] Исключить недавно показанные задания
- [ ] Тесты IRT-логики

---

## Этап 6 — Горячий путь (сквозной)

Соединить сервисы в единый синхронный pipeline.

- [ ] `services/gateway/` — отдельный FastAPI-сервис, оркестрирует горячий путь
- [ ] Эндпоинт в gateway: `POST /next-task` (принимает `student_id`)
  - [ ] Профиль → Retrieval → Ranking → Task Bank → вернуть задание
- [ ] `POST /submit-answer` — принять ответ ученика
  - [ ] Обновить mastery в Profile
  - [ ] Опубликовать событие `answer_submitted` в Kafka
- [ ] E2E-тест: ученик решает 5 заданий подряд, mastery растёт
- [ ] Замер latency горячего пути (цель < 500ms)

---

## Этап 7 — Холодный путь (Kafka)

Асинхронная обработка событий после ответа ученика.

- [ ] Kafka топики: `answer_submitted`, `mastery_updated`
- [ ] Консьюмер в сервисе аналитики: сохранять `Interaction` в `interactions`
- [ ] Консьюмер планировщика: заглушка для будущего учебного плана
- [ ] Тест: событие публикуется → консьюмер обрабатывает

---

## Этап 8 — Cold Start / Placement Assessment

Онбординг нового ученика.

- [ ] `POST /students` принимает `grade` → инициализировать mastery
  - [ ] KC из классов < grade → `mastery = 0.5`
  - [ ] KC текущего и выше → `mastery = 0.0`
- [ ] `GET /diagnostic-test?student_id={id}` → 10–15 диагностических вопросов
  - [ ] По 1–2 KC на раздел текущего класса
- [ ] `POST /diagnostic-test/submit` → скорректировать mastery по результатам
- [ ] Тест: после диагностики mastery отражает реальный уровень

---

## Этап 9 — Интеграция и запуск

- [ ] `docker-compose.full.yml` — все сервисы в контейнерах
- [ ] Health check эндпоинты (`GET /health`) во всех сервисах
- [ ] `README.md` — как запустить систему локально
- [ ] Ручное smoke-тестирование полного флоу

---

## Отложено (не в плане)

- Сервис генерации заданий (LLM)
- IRT калибровка новых заданий
- Exploration / Thompson Sampling
- Redis кеш mastery
- Evaluation pipeline (learning gain, retention)
