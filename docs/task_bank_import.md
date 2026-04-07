# Импорт заданий в `task_bank`

Сейчас в репозитории есть два фактических ограничения:

- `exam_data/` содержит только пустую структуру каталогов без самих заданий.
- Граф в [services/graph/seed.py](/home/alex/coding/learnity/services/graph/seed.py) описывает только математику 5–9, а в `exam_data/` лежат `istoriya`, `informatika`, `obshchestvoznanie`.

Из этого следует:

- прямо сейчас невозможно автоматически собрать полноценную таблицу заданий;
- даже после появления текстов задач текущий граф тем не подойдет для нематематических предметов.

## Что добавлено

Есть утилита [tools/build_task_bank_import.py](/home/alex/coding/learnity/tools/build_task_bank_import.py). Она умеет:

- создать `staging`-файл из структуры `exam_data/`;
- провалидировать строки под формат из [task_bank_schema.md](/home/alex/coding/learnity/task_bank_schema.md);
- выгрузить `CSV` и `SQL` для загрузки в PostgreSQL.

## Workflow

1. Сгенерировать шаблон строк из `exam_data/`:

```bash
python tools/build_task_bank_import.py bootstrap
```

2. Открыть `exam_data/task_bank_staging.jsonl` и заполнить для каждой строки:

- `tutor_id`
- `subject_id`
- `body`
- `answer`
- `solution`
- `topics`

3. Если у тебя уже есть привязка к `kc_id` из графа, можно временно хранить её в `topic_kc_ids`, но в экспорт уйдут только integer ids из `topics`.

4. Проверить заполнение:

```bash
python tools/build_task_bank_import.py validate
```

5. Собрать артефакты импорта:

```bash
python tools/build_task_bank_import.py export --format both
```

На выходе получишь:

- `task_bank_import.csv`
- `task_bank_import.sql`

## Формат staging JSONL

Одна строка = одно задание.

```json
{
  "source_path": "oge/informatika-oge-67/15",
  "exam_kind": "oge",
  "package_slug": "informatika-oge-67",
  "task_number": "15",
  "tutor_id": "00000000-0000-0000-0000-000000000000",
  "subject_id": "00000000-0000-0000-0000-000000000000",
  "task_type": "exam",
  "answer_type": "open",
  "difficulty": "medium",
  "body": "Текст задания",
  "answer": ["42"],
  "solution": ["Шаг 1", "Шаг 2"],
  "topics": [101, 205],
  "topic_kc_ids": ["kc_linear_eq_1var"],
  "is_public": true,
  "is_generated": false,
  "deadline": null
}
```

## Что нужно решить дальше

- Либо добавить реальные математические задания в `exam_data/`, если рекомендационная система пока только по математике.
- Либо расширить граф тем и `common_topics.yaml` под историю, информатику и обществознание.
- Либо держать `task_bank` мультипредметным, но тогда `topics` должны ссылаться на разные предметные таксономии, а не только на текущий math-граф.
