# Learnity — Kafka Integration Contracts

Интеграция backend (Go) ↔ Learnity (Python) через Kafka.

---

## Контракт #1: Студент зарегистрирован

**Topic:** `learnity.student.registered`  
**Направление:** backend → Learnity  
**Когда:** backend создаёт нового пользователя с ролью `student`

```json
{
  "event_type": "STUDENT_REGISTERED",
  "student_id": "uuid",
  "grade": 8,
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `student_id` | UUID | `users.id` из базы бекэнда |
| `grade` | int 5–11 | `users.grade` — класс ученика |

**Что делает Learnity:** инициализирует студента + cold_start mastery по всем KC исходя из grade.

---

## Контракт #2: Студент ответил на задание

**Topic:** `learnity.task.answered`  
**Направление:** backend → Learnity  
**Когда:** backend создаёт `task_submission` (студент отправил ответ)

```json
{
  "event_type": "TASK_ANSWERED",
  "student_id": "uuid",
  "task_id": "uuid",
  "score": 0.85,
  "hints_used": 1,
  "topic": "linear_equations",
  "subject": "algebra",
  "difficulty": "3",
  "recommendation_source": "linucb",
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `student_id` | UUID | Кто отвечал |
| `task_id` | UUID | `task_bank.id` из бекэнда |
| `score` | float 0.0–1.0 | Правильность ответа: 1.0 верно, 0.0 неверно |
| `hints_used` | int | Сколько подсказок использовал |
| `topic` | str | `task_bank.topic` — тема задания (= kc_id в Learnity) |
| `subject` | str | Предмет (algebra, geometry...) |
| `difficulty` | str "1"–"5" | `task_bank.difficulty` |
| `recommendation_source` | str? | Источник рекомендации — передаётся обратно из ответа Learnity на запрос задания |

**Что делает Learnity:** обновляет mastery студента по теме, обновляет бандит-модель.

---

## Контракт #3: Запрос следующего задания

**Topic:** `learnity.task.request`  
**Направление:** backend → Learnity  
**Когда:** студент запрашивает следующее задание

```json
{
  "event_type": "TASK_REQUESTED",
  "request_id": "uuid",
  "student_id": "uuid",
  "subject": "algebra",
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `request_id` | UUID | Уникальный ID запроса — используется для сопоставления с ответом |
| `student_id` | UUID | Кому рекомендуем |
| `subject` | str | Предмет в рамках которого делается рекомендация |

**Что делает Learnity:** выбирает задание через LinUCB+ZPD, публикует ответ в `learnity.task.recommended`.

---

## Контракт #4: Рекомендация задания

**Topic:** `learnity.task.recommended`  
**Направление:** Learnity → backend  
**Когда:** в ответ на `TASK_REQUESTED`

```json
{
  "event_type": "TASK_RECOMMENDED",
  "request_id": "uuid",
  "student_id": "uuid",
  "task_id": "uuid",
  "topic": "linear_equations",
  "recommendation_source": "linucb",
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:05Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `request_id` | UUID | Тот же ID что пришёл в запросе — для сопоставления |
| `student_id` | UUID | Кому рекомендовано |
| `task_id` | UUID | `task_bank.id` — backend достаёт задание из своего банка |
| `topic` | str | Тема задания |
| `recommendation_source` | str | Почему выбрано: `plan` — из учебного плана, `zpd` — по ZPD, `explore` — исследование |

**Что делает backend:** достаёт задание из task_bank по `task_id`, показывает студенту. Сохраняет `recommendation_source` чтобы передать обратно в контракте #2 при ответе.

---

## Контракт #5: Запрос mastery

**Topic (запрос):** `learnity.mastery.request`  
**Topic (ответ):** `learnity.mastery.response`  
**Направление:** backend → Learnity → backend  
**Когда:** backend хочет показать прогресс ученика (профиль, карта навыков)

**Запрос:**
```json
{
  "event_type": "MASTERY_REQUESTED",
  "request_id": "uuid",
  "student_id": "uuid",
  "skill_keys": ["linear_equations", "quadratic_equations", "fractions"],
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

**Ответ:**
```json
{
  "event_type": "MASTERY_RESPONSE",
  "request_id": "uuid",
  "student_id": "uuid",
  "levels": [
    { "skill_key": "linear_equations", "level": 0.74 },
    { "skill_key": "quadratic_equations", "level": 0.41 },
    { "skill_key": "fractions", "level": 0.88 }
  ],
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:01Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `student_id` | UUID | Чей mastery запрашивается |
| `skill_keys` | list[str] | Темы по которым нужен mastery. Если пустой — возвращаем все |
| `levels[].skill_key` | str | Тема |
| `levels[].level` | float 0.0–1.0 | Текущий mastery |

---

## Контракт #6: Teacher alert

**Topic:** `learnity.alert.teacher`  
**Направление:** Learnity → backend  
**Когда:** студент застрял на теме и нужно вмешательство учителя

```json
{
  "event_type": "TEACHER_ALERT",
  "student_id": "uuid",
  "alert_type": "plateau",
  "skill_key": "linear_equations",
  "mastery_at_alert": 0.38,
  "tasks_spent": 54,
  "message": "Студент не прогрессирует по теме 54 задания подряд",
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `student_id` | UUID | Кто застрял |
| `alert_type` | str | `plateau` — нет прогресса; `replan_requested` — план неэффективен |
| `skill_key` | str | На какой теме застрял |
| `mastery_at_alert` | float | Уровень mastery в момент алерта |
| `tasks_spent` | int | Сколько заданий потрачено без прогресса |
| `message` | str | Описание для учителя |

**Что делает backend:** создаёт уведомление для tutor-а студента.

---

## Контракт #7: Создание учебного плана

**Topic (запрос):** `learnity.plan.request`  
**Topic (ответ):** `learnity.plan.created`  
**Направление:** backend → Learnity → backend  
**Когда:** tutor хочет создать учебный план для студента

**Запрос:**
```json
{
  "event_type": "PLAN_REQUESTED",
  "request_id": "uuid",
  "student_id": "uuid",
  "mode": "target_mastery",
  "target_skill_key": "quadratic_equations",
  "mastery_threshold": 0.80,
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:00Z"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `request_id` | UUID | Для сопоставления с ответом |
| `student_id` | UUID | Для кого план |
| `mode` | str | `target_mastery` — довести до порога по конкретной теме; `coverage` — охватить список тем |
| `target_skill_key` | str? | Целевая тема (только для `target_mastery`) |
| `mastery_threshold` | float | Порог освоения. По умолчанию 0.80 |

**Ответ:**
```json
{
  "event_type": "PLAN_CREATED",
  "request_id": "uuid",
  "student_id": "uuid",
  "plan_id": "uuid",
  "steps_count": 5,
  "schema_version": "v1",
  "created_at": "2026-04-07T12:00:01Z"
}
```

**Что делает backend:** сохраняет `plan_id` для дальнейших запросов прогресса.
