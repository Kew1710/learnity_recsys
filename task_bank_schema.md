# Схема таблицы task_bank

## Столбцы

| Столбец | Тип | Пример |
|---|---|---|
| `id` | `uuid` | генерируется автоматически |
| `tutor_id` | `uuid` | uuid репетитора-владельца |
| `subject_id` | `uuid` | uuid предмета |
| `task_type` | enum | `'homework'` |
| `answer_type` | enum | `'single_choice'` |
| `difficulty` | text | `'medium'` |
| `body` | text | `'Решите уравнение: 2x + 3 = 7'` |
| `answer` | `text[]` | `ARRAY['x = 2']` |
| `solution` | `text[]` | `ARRAY['Переносим 3: 2x = 4', 'Делим на 2: x = 2']` |
| `topics` | `integer[]` | `ARRAY[12, 34]` |
| `is_public` | boolean | `true` |
| `is_generated` | boolean | `true` |
| `deadline` | timestamptz | NULL (необязательно) |

## Допустимые значения enum

**task_type:** `diagnostic`, `homework`, `practice`, `exam`, `project`

**answer_type:** `single_choice`, `multiple_choice`, `open`, `numeric`, `code`

**difficulty:** `very_easy`, `easy`, `medium`, `hard`, `very_hard`

## Про поля-массивы

- `answer` — все варианты правильного ответа (можно несколько для разных формулировок)
- `solution` — шаги решения по порядку, каждый шаг — отдельная строка массива
- `topics` — список целых чисел, обозначающих темы задания (произвольные id, например 101, 205)

## Пример INSERT

```sql
INSERT INTO task_bank (tutor_id, subject_id, task_type, answer_type, difficulty, body, answer, solution, topics, is_public, is_generated)
VALUES (
  '<tutor_uuid>',
  '<subject_uuid>',
  'homework',
  'single_choice',
  'medium',
  'Решите уравнение: 2x + 3 = 7',
  ARRAY['x = 2'],
  ARRAY['Переносим 3 в правую часть: 2x = 4', 'Делим обе части на 2: x = 2'],
  ARRAY[101, 205],
  true,
  true
);
```
