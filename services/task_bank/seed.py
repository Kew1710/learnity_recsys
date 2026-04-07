"""
Синтетический seed банка заданий.

Генерирует 100 stub-заданий на каждую KC из графа знаний.
Задания не содержат реального текста — только метаданные:
  - к какой KC относятся
  - базовая сложность (равномерно [0.05, 0.95])
  - тип задания: procedural / conceptual / word_problem / mixed
  - n_steps: количество шагов до ответа

Запуск: python -m services.task_bank.seed

Когда добавишь реальные задания — они просто дополнят таблицу рядом
с этими стабами. Стабы можно оставить или удалить через DELETE WHERE source='stub'.
"""

import asyncio
import uuid

from shared.db import AsyncSessionLocal
from services.graph.seed import KCS
from .models import TaskModel, PartModel

# 100 уровней сложности равномерно по всему диапазону [0.05, 0.95].
TASK_DIFFICULTIES = [round(0.05 + i * (0.90 / 99), 3) for i in range(100)]

# Распределение типов на 100 заданий: 40 procedural, 30 conceptual, 20 word_problem, 10 mixed.
# Повторяем паттерн чтобы покрыть весь диапазон сложности для каждого типа.
_TYPE_PATTERN = (
    ["procedural"] * 4 + ["conceptual"] * 3 + ["word_problem"] * 2 + ["mixed"] * 1
)  # 10 шт, повторяется 10 раз → 100 заданий

# n_steps зависит от типа: процедурные проще (1-2 шага), смешанные сложнее (3-5).
_TYPE_N_STEPS = {
    "procedural":  (1, 2),
    "conceptual":  (1, 3),
    "word_problem":(2, 4),
    "mixed":       (3, 5),
}


def _build_stub(kc: dict, task_index: int) -> TaskModel:
    kc_id = kc["kc_id"]
    kc_name = kc["name"]
    grade = kc["grade_introduced"]
    difficulty = TASK_DIFFICULTIES[task_index]
    task_num = task_index + 1

    task_type = _TYPE_PATTERN[task_index % len(_TYPE_PATTERN)]
    n_steps_min, n_steps_max = _TYPE_N_STEPS[task_type]
    # Детерминировано из task_index чтобы seed был воспроизводимым
    n_steps = n_steps_min + (task_index // len(_TYPE_PATTERN)) % (n_steps_max - n_steps_min + 1)

    task = TaskModel(
        task_id=uuid.uuid4(),
        grade_min=grade,
        source="stub",
    )
    part = PartModel(
        task_id=task.task_id,
        part_id="p1",
        description=f"[STUB] {kc_name} — {task_type} задача {task_num}",
        primary_kcs=[kc_id],
        secondary_kcs=[],
        answer_type="numeric",
        correct_answer=None,
        tolerance=None,
        irt_difficulty=difficulty,
        irt_discrimination=None,
        irt_guessing=None,
        task_type=task_type,
        n_steps=n_steps,
        scaffolding_steps=[],
        distractors_map={},
    )
    task.parts = [part]
    return task


async def seed() -> None:
    async with AsyncSessionLocal() as session:
        # Удаляем старые стабы перед заливкой новых
        from sqlalchemy import text
        await session.execute(text("DELETE FROM tasks WHERE source='stub'"))
        await session.commit()

        tasks = [
            _build_stub(kc, i)
            for kc in KCS
            for i in range(len(TASK_DIFFICULTIES))
        ]

        session.add_all(tasks)
        await session.commit()
        print(f"Залито {len(tasks)} stub-заданий ({len(KCS)} KC × {len(TASK_DIFFICULTIES)} вариантов, сложность [0.05, 0.95]).")


if __name__ == "__main__":
    asyncio.run(seed())
