# Интеграция Backend ↔ Learnity

Этот документ написан для backend-разработчика, который будет связывать Go backend с Learnity и выкатывать всё это в рабочее окружение.

Ниже описано:

- что уже есть в Learnity;
- какие сервисы за что отвечают;
- как сервисы общаются между собой;
- какие хранилища используются;
- где граница между backend и Learnity;
- как это поднимать локально;
- как это раскладывать в deploy.

## 1. Что такое Learnity

Learnity в этом репозитории это отдельный Python-контур адаптивной рекомендации.

Его зона ответственности:

- хранить mastery ученика по `kc_id`;
- хранить и обслуживать граф знаний;
- выбирать следующую тему и следующее задание;
- вести bandit/IRT/BKT-логику;
- строить учебные планы;
- отдавать teacher alerts и ответы по mastery.

Это не замена backend.
Backend остаётся источником пользователей, курсов, UI, авторизации, question/task bank интерфейсов, submission flow и teacher/student бизнес-логики.

## 2. Сервисы Learnity

### `gateway` — [services/gateway/main.py](/home/alex/coding/learnity/services/gateway/main.py)

Точка входа в Learnity.

Что делает:

- слушает Kafka-события от backend;
- публикует ответы обратно в Kafka;
- умеет работать как HTTP gateway поверх внутренних сервисов;
- проксирует backend через `/backend/*` для локальных integration-тестов.

Локальный порт:

- `8005`

### `profile` — [services/profile/main.py](/home/alex/coding/learnity/services/profile/main.py)

Хранит профиль ученика и mastery.

Что делает:

- создаёт ученика;
- делает cold-start mastery;
- обновляет mastery после ответа;
- хранит seen tasks;
- хранит learning plans и часть аналитики взаимодействий.

Локальный порт:

- `8001`

### `graph` — [services/graph/main.py](/home/alex/coding/learnity/services/graph/main.py)

Сервис графа знаний.

Что делает:

- хранит KC и prerequisite graph;
- отдаёт ZPD-кандидатов;
- отдаёт prereq-цепочки и KC-метаданные.

Локальный порт:

- `8002`

### `task_bank` — [services/task_bank/main.py](/home/alex/coding/learnity/services/task_bank/main.py)

Сервис банка заданий, который retrieval использует для выдачи задач по `kc_id`.

Что делает:

- хранит задания с `primary_kcs` и `secondary_kcs`;
- отдаёт задачи по `kc_id`;
- хранит IRT/difficulty/task_type/n_steps.

Локальный порт:

- `8003`

Важно:

- сейчас retrieval в этом репозитории ходит именно в Python `task_bank`, а не напрямую в Go backend.
- если в production задания физически хранятся только в backend, есть 2 варианта:
  1. синхронизировать задания backend → Learnity task_bank;
  2. переписать retrieval, чтобы он брал задания из backend API вместо `services/task_bank`.

### `retrieval` — [services/retrieval/main.py](/home/alex/coding/learnity/services/retrieval/main.py)

Сервис выбора следующего задания.

Что делает:

- получает mastery и seen tasks;
- запрашивает ZPD из graph;
- выбирает `kc_id`;
- запрашивает задания из task_bank;
- применяет LinUCB/IRT/subject rotation;
- возвращает `task_id`, `kc_id`, `recommendation_source`.

Локальный порт:

- `8004`

### `macro` — [services/macro/main.py](/home/alex/coding/learnity/services/macro/main.py)

Сервис учебных планов.

Что делает:

- создаёт план на основе target KC или coverage mode;
- отслеживает прогресс по плану;
- переключает шаги;
- может отправлять teacher alerts и пересобирать план.

Локальный порт:

- `8006`

## 3. Как Learnity общается внутри себя

### HTTP между сервисами

Внутри Learnity межсервисное общение сейчас HTTP-based:

- `gateway -> profile`, `gateway -> retrieval`, `gateway -> graph`, `gateway -> macro`
- `retrieval -> profile`, `retrieval -> graph`, `retrieval -> task_bank`
- `macro -> profile`, `macro -> graph`

Клиенты:

- [services/gateway/clients.py](/home/alex/coding/learnity/services/gateway/clients.py)
- [services/retrieval/clients.py](/home/alex/coding/learnity/services/retrieval/clients.py)

### Kafka между backend и Learnity

Граница backend ↔ Learnity идёт через Kafka.

Контракты описаны в:

- [docs/kafka_contracts.md](/home/alex/coding/learnity/docs/kafka_contracts.md)

Основные inbound топики backend → Learnity:

- `learnity.student.registered`
- `learnity.task.answered`
- `learnity.task.request`
- `learnity.mastery.request`
- `learnity.plan.request`

Основные outbound топики Learnity → backend:

- `learnity.task.recommended`
- `learnity.mastery.response`
- `learnity.plan.created`
- `learnity.alert.teacher`

Producer:

- [services/gateway/kafka_producer.py](/home/alex/coding/learnity/services/gateway/kafka_producer.py)

Consumer:

- [services/gateway/kafka_consumer.py](/home/alex/coding/learnity/services/gateway/kafka_consumer.py)

## 4. Хранилища данных

### Хранилища Learnity

Learnity использует:

- PostgreSQL на `5432`
- Neo4j на `7688`
- Kafka на `9092`

Инфраструктура описана в:

- [docker-compose.yml](/home/alex/coding/learnity/docker-compose.yml)

PostgreSQL Learnity общий для Python-сервисов:

- `profile`
- `task_bank`
- `retrieval` bandit tables
- `macro` learning plans

Подключение по умолчанию:

- `postgresql+asyncpg://learnity:learnity@localhost:5432/learnity`

См.:

- [shared/db.py](/home/alex/coding/learnity/shared/db.py)

Neo4j используется только графом знаний.

Kafka используется как шина интеграции с backend.

### Хранилища backend

Backend поднимает отдельно:

- PostgreSQL на `5433`
- Redis на `6379`

См.:

- [docker-compose.backend.yml](/home/alex/coding/learnity/docker-compose.backend.yml)
- [backend/learnity/.env.example](/home/alex/coding/learnity/backend/learnity/.env.example)

### Есть ли общая база между backend и Learnity

Сейчас правильная модель такая:

- общей БД между backend и Learnity быть не должно;
- интеграция должна идти через Kafka;
- опционально допустим HTTP к backend API или к Learnity gateway, но не прямой доступ к чужим таблицам.

Единственный реально общий смысловой слой:

- `student_id`
- `task_id`
- `kc_id` / `topic`
- mapping `kc_id -> common_topic_id`

## 5. Общие идентификаторы и что должно совпадать

### `student_id`

`student_id` должен быть одинаковым в backend и Learnity.

Источник истины:

- backend

Как попадает в Learnity:

- событие `STUDENT_REGISTERED`

### `task_id`

`task_id` должен быть одинаковым в backend и в том task bank, из которого retrieval выбирает задачу.

Это критично, потому что:

- Learnity возвращает в `TASK_RECOMMENDED` именно `task_id`;
- backend потом по этому `task_id` должен достать задание и показать ученику.

Если Learnity task_bank и backend task_bank разные, нужно заранее решить стратегию:

1. Единый источник `task_id`, общий для обеих сторон.
2. Регулярная синхронизация задач из backend в Learnity.
3. Или отказ от Python task_bank и прямой retrieval against backend.

### `kc_id`

Внутренний канонический ключ темы в Learnity.

Используется в:

- graph
- profile mastery
- retrieval
- learning plans
- Kafka topic field в recsys-контракте

Менять его на integer ids из `common_topics.yaml` не нужно.

### `common_topic_id`

Это backend/analytics layer.

Мы уже собрали mapping:

- простой dict: [docs/kc_topic_mapping_simple.json](/home/alex/coding/learnity/docs/kc_topic_mapping_simple.json)
- review file: [docs/kc_topic_mapping_review.json](/home/alex/coding/learnity/docs/kc_topic_mapping_review.json)

Использование:

- Learnity живёт на `kc_id`
- backend metadata/analytics/diagnostics живут на `common_topic_id`
- при импорте задания нужно проставлять оба слоя:
  - `topic = kc_id` для Learnity/recsys-контуров
  - `metadata.skill_keys = [common_topic_id, ...]` для backend

## 6. Рекомендуемая схема интеграции backend ↔ Learnity

### Минимальная рабочая схема

1. Backend создаёт ученика.
2. Backend публикует `STUDENT_REGISTERED`.
3. Learnity создаёт профиль и cold-start mastery.
4. Когда ученику нужно следующее задание, backend публикует `TASK_REQUESTED`.
5. Learnity публикует `TASK_RECOMMENDED`.
6. Backend достаёт задачу по `task_id` и показывает её пользователю.
7. После ответа backend публикует `TASK_ANSWERED`.
8. Learnity обновляет mastery и bandit-модель.

### Для progress UI

1. Backend публикует `MASTERY_REQUESTED`.
2. Learnity отвечает `MASTERY_RESPONSE`.
3. Backend рисует skill map / progress bars.

### Для teacher planning

1. Backend публикует `PLAN_REQUESTED`.
2. Learnity создаёт план.
3. Learnity отвечает `PLAN_CREATED`.
4. Backend показывает план в teacher UI.

## 7. Что нужно от backend, чтобы интеграция была рабочей

### Обязательно

- публиковать Kafka-события из `docs/kafka_contracts.md`;
- использовать один и тот же `student_id`;
- гарантировать, что `task_id`, который вернул Learnity, реально существует в backend question/task bank;
- хранить и передавать обратно `recommendation_source`;
- в заданиях иметь `topic = kc_id`, если backend публикует ответы в Learnity через Kafka `topic`.

### Очень желательно

- хранить `metadata.skill_keys` как integer ids из `common_topics`;
- использовать mapping из [docs/kc_topic_mapping_simple.json](/home/alex/coding/learnity/docs/kc_topic_mapping_simple.json);
- логировать `request_id` для request/response Kafka-пар;
- не читать данные Learnity напрямую из его Postgres.

## 8. Локальный запуск

### 8.1 Поднять инфраструктуру Learnity

```bash
make up
```

Это поднимет:

- Postgres `:5432`
- Neo4j `:7688`
- Kafka `:9092`

### 8.2 Установить Python-зависимости

```bash
make install
```

### 8.3 Применить миграции Learnity

```bash
make migrate
```

### 8.4 Залить seed-данные

```bash
make seed
```

Что делает:

- seed графа в Neo4j
- invalidate cache graph
- seed Python task_bank

### 8.5 Поднять Python-сервисы

```bash
make dev
```

Порты:

- Gateway `8005`
- Profile `8001`
- Graph `8002`
- TaskBank `8003`
- Retrieval `8004`
- Macro `8006`

Проверка:

```bash
make dev-status
```

### 8.6 Поднять backend-инфраструктуру

```bash
make backend-up
make backend-migrate
```

### 8.7 Запустить backend

```bash
make backend-run
```

По умолчанию backend поднимется на:

- `http://localhost:8080`

### 8.8 Проверить Kafka-интеграцию

Есть готовый smoke-тест:

```bash
.venv/bin/python tools/test_kafka_integration.py
```

Если всё работает, должен пройти сценарий:

- `TASK_REQUESTED`
- `TASK_RECOMMENDED`
- `TASK_ANSWERED`
- `MASTERY_REQUESTED`

## 9. Как это деплоить

### Вариант A. Быстрый и понятный

Разнести на 3 логических блока:

1. Backend stack
2. Learnity infra
3. Learnity app services

#### Backend stack

- Go backend
- backend Postgres
- Redis

#### Learnity infra

- Learnity Postgres
- Neo4j
- Kafka

#### Learnity app services

- gateway
- profile
- graph
- task_bank
- retrieval
- macro

### Вариант B. Один docker-compose / k8s namespace

Можно держать всё в одном окружении, но логически всё равно разделять:

- backend DB не шарить с Learnity;
- Learnity DB не шарить с backend;
- коммуникацию между ними держать через Kafka.

## 10. Какие env нужны в production

### Learnity

Минимально:

- `DATABASE_URL`
- `NEO4J_URI`
- `KAFKA_BOOTSTRAP_SERVERS`
- `PROFILE_URL`
- `GRAPH_URL`
- `TASK_BANK_URL`
- `RETRIEVAL_URL`
- `MACRO_URL`
- `BACKEND_URL`

### Backend

См.:

- [backend/learnity/.env.example](/home/alex/coding/learnity/backend/learnity/.env.example)

Для recsys-интеграции критично:

- `KAFKA_BOOTSTRAP_SERVERS`
- `RECSYS_KAFKA_GROUP_ID`
- `RECSYS_KAFKA_TOPIC_STUDENT_REGISTERED`
- `RECSYS_KAFKA_TOPIC_TASK_ANSWERED`
- `RECSYS_KAFKA_TOPIC_TASK_REQUEST`
- `RECSYS_KAFKA_TOPIC_TASK_RECOMMENDED`
- `RECSYS_KAFKA_TOPIC_MASTERY_REQUEST`
- `RECSYS_KAFKA_TOPIC_MASTERY_RESPONSE`
- `RECSYS_KAFKA_TOPIC_ALERT_TEACHER`
- `RECSYS_KAFKA_TOPIC_PLAN_REQUEST`
- `RECSYS_KAFKA_TOPIC_PLAN_CREATED`

## 11. Практические решения, которые нужно принять до production

### Решение 1. Где живёт source of truth для task bank

Нужно зафиксировать одно из двух:

- backend — source of truth, а Learnity task_bank это read-model / mirror;
- Learnity task_bank — source of truth для recommendation, backend только показывает задачу по тому же `task_id`.

Без этого будет рассинхрон по `task_id`.

### Решение 2. Как в backend хранить тему задачи

Рекомендуемый вариант:

- `topic = kc_id`
- `metadata.skill_keys = [common_topic_id, ...]`

Так backend получает и recsys-связку, и нормальную аналитику по своим integer ids.

### Решение 3. Нужен ли direct HTTP path кроме Kafka

Рекомендуемая граница:

- asynchronous события и ответы через Kafka;
- direct HTTP использовать только для локальной диагностики и smoke-тестов.

## 12. Короткий итог

В текущем состоянии Learnity это отдельный recommendation engine со своими сервисами и своим storage-контуром.

Правильная интеграция с backend:

- не через общую БД;
- не через подмену `kc_id`;
- а через Kafka-контракты + согласованные `student_id`/`task_id` + mapping `kc_id -> common_topic_id`.

Если нужно связать систему быстро и без лишней хрупкости, то минимальный production-safe путь такой:

1. backend остаётся владельцем пользователей и UI;
2. Learnity остаётся владельцем mastery/graph/recommendation/plans;
3. topic для recsys хранится как `kc_id`;
4. analytics-темы backend живут в `metadata.skill_keys`;
5. обмен идёт через Kafka.
