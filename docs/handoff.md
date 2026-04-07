# Learnity — Handoff для интеграции с backend

> Состояние на 2026-04-07

---

## Что реализовано

### Хранение знаний студента (mastery)
- Profile service (PostgreSQL) хранит mastery по каждому KC для каждого студента
- BKT (Bayesian Knowledge Tracing) + IRT обновляет mastery после каждого ответа
- Cold-start при регистрации: mastery инициализируется по классу (grade) студента
- Decay: `probability_effective` снижается если студент давно не практиковал тему

### Рекомендация заданий
- Retrieval service: LinUCB + ZPD выбирает KC → selector возвращает задание по kc_id
- Учитывает активный учебный план (приоритизирует KC из плана)
- Subject rotation: не выдаёт подряд одну и ту же тему
- A/B эксперименты: LinUCB vs Baseline (random)

### Kafka-интеграция с backend (Go)
Все 7 контрактов задокументированы в `docs/kafka_contracts.md`. Реализованы:

| # | Топик | Статус |
|---|---|---|
| 1 | `learnity.student.registered` → инициализация профиля | ✅ |
| 2 | `learnity.task.answered` → обновление mastery | ✅ |
| 3 | `learnity.task.request` → запрос задания | ✅ |
| 4 | `learnity.task.recommended` → ответ с заданием | ✅ |
| 5 | `learnity.mastery.request/response` → получение mastery | ✅ |
| 6 | `learnity.alert.teacher` → алерт если студент застрял | ✅ (Learnity публикует; backend должен консьюмить) |
| 7 | `learnity.plan.request/response` → создание учебного плана | ✅ |

### Учебные планы
- Macro service создаёт планы по запросу (`target_mastery` или `coverage`)
- При достижении порога mastery — план автоматически переходит к следующему шагу
- Если студент застрял — публикует `TEACHER_ALERT` в `learnity.alert.teacher`
- Замена плана: создание нового плана деактивирует старый

---

## Что НЕ реализовано (нужно сделать на стороне backend)

| Пункт | Что нужно |
|---|---|
| Получение mastery через REST | `GET /api/v1/students/:id/skill-map` сейчас читает из своей БД. Нужно либо консьюмить `learnity.mastery.response` и кэшировать, либо добавить прямой HTTP-запрос к Learnity |
| Teacher alerts | Backend не консьюмит `learnity.alert.teacher`. Нужно добавить consumer и создавать уведомления для tutor-а |
| Замена плана через UI | Нет HTTP-эндпоинта для замены плана (только Kafka). Backend может просто заново опубликовать `PLAN_REQUESTED` — Learnity заменит старый план |

---

## Как запустить

### Инфраструктура (Kafka, Neo4j, PostgreSQL)
```bash
cd /home/alex/coding/learnity
docker compose up -d          # Neo4j :7688, Kafka :9092, PG :5432
```

### Backend инфраструктура (отдельная PG + Redis)
```bash
make backend-up               # PG :5433, Redis :6379
make backend-migrate          # Применить миграции backend
```

### Learnity сервисы (Python)
```bash
make seed                     # Один раз: граф знаний + банк заданий
make dev                      # Запуск всех 6 сервисов
# Gateway   → http://localhost:8005
# Profile   → http://localhost:8001
# Graph     → http://localhost:8002
# TaskBank  → http://localhost:8003
# Retrieval → http://localhost:8004
# Macro     → http://localhost:8006
```

### Backend (Go)
```bash
cd backend/learnity && ./run-local.sh   # :8080
# Swagger: http://localhost:8080/swagger/index.html
```

### Интеграционный UI
```
http://localhost:8005/static/integration-test.html
```
Позволяет: зарегистрировать студента → войти → подключиться по WS → запросить задание → запросить mastery.

---

## Структура сервисов

```
services/
  gateway/      — точка входа, прокси к backend, Kafka producer/consumer
  profile/      — хранение mastery, BKT, cold-start
  retrieval/    — выбор задания, LinUCB, ZPD, A/B
  graph/        — граф знаний (KC, пре-реквизиты), Neo4j
  task_bank/    — банк заданий, PostgreSQL (14 200 заданий, 5–11 класс)
  macro/        — учебные планы, teacher alerts, эскалация
  clustering/   — кластеризация студентов для cold-start
```

---

## Важные детали

- **task_bank** содержит задания с KC в поле `topic` — это же `kc_id` в Learnity. Они должны совпадать.
- **recommendation_source** из ответа Learnity нужно сохранить и передать обратно в `TASK_ANSWERED` — используется для обновления bandit-модели.
- **WS-подключение** к backend: `ws://localhost:8080/api/v1/recsys/ws?access_token=<JWT>`
- Kafka топики создаются автоматически при первом запуске gateway.

---

## Ссылки на документацию
- `docs/kafka_contracts.md` — полные JSON-схемы всех 7 контрактов
- `docs/integration.md` — поля запросов/ответов (таблицы)
- `docs/system_architecture.md` — архитектура системы
