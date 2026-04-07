"""Вычисление MicroSummary из bandit_log для конкретного ученика и KC."""

from __future__ import annotations

import math
import uuid

import sqlalchemy as sa

from shared.db import AsyncSessionLocal

# Сколько последних записей анализируем для summary
SUMMARY_WINDOW = 20


async def compute_micro_summary(
    student_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    tasks_spent: int,
) -> dict:
    """
    Считает MicroSummary по последним SUMMARY_WINDOW записям bandit_log для KC.

    Поля:
      student_id, kc_id
      mastery_current     — текущий mastery (из профиля)
      velocity            — средний прирост reward за последние N заданий
      frustration_count   — количество подряд идущих reward < 0.3
      avg_score           — средний reward
      hint_rate           — доля заданий с hints_used > 0 (из interactions)
      irt_residual        — среднее |P_correct - reward| (насколько IRT модель ошибается)
      tasks_spent         — сколько заданий выдано по этой KC в текущем шаге плана
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT bl.reward, bl.context_vector
                FROM bandit_log bl
                WHERE bl.student_id = :student_id
                  AND bl.kc_id = :kc_id
                  AND bl.reward IS NOT NULL
                ORDER BY bl.recommended_at DESC
                LIMIT :window
            """),
            {"student_id": student_id, "kc_id": kc_id, "window": SUMMARY_WINDOW},
        )).fetchall()

    rewards = [float(r[0]) for r in rows]

    if not rewards:
        return _empty_summary(student_id, kc_id, mastery_current, tasks_spent)

    avg_score = sum(rewards) / len(rewards)

    # velocity: наклон линейной регрессии reward по времени (упрощённо — разница первой и последней половины)
    mid = len(rewards) // 2
    if mid > 0:
        early = sum(rewards[mid:]) / (len(rewards) - mid)   # более ранние (DESC порядок)
        late = sum(rewards[:mid]) / mid                      # более поздние
        velocity = late - early
    else:
        velocity = 0.0

    # frustration: последовательные награды < 0.3 с начала (последние по времени)
    frustration_count = 0
    for r in rewards:
        if r < 0.3:
            frustration_count += 1
        else:
            break

    # irt_residual: среднее |P_correct(mastery, difficulty) - reward|
    # context_vector[0] = mastery_kc, для простоты используем mastery_current
    # difficulty мы не храним напрямую в bandit_log, поэтому используем 0 как fallback
    irt_residual = 0.0  # Phase 3 подключит реальную difficulty из task parts

    return {
        "student_id": str(student_id),
        "kc_id": kc_id,
        "mastery_current": mastery_current,
        "velocity": round(velocity, 4),
        "frustration_count": frustration_count,
        "avg_score": round(avg_score, 4),
        "irt_residual": irt_residual,
        "tasks_spent": tasks_spent,
        "sample_size": len(rewards),
    }


def _empty_summary(
    student_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    tasks_spent: int,
) -> dict:
    return {
        "student_id": str(student_id),
        "kc_id": kc_id,
        "mastery_current": mastery_current,
        "velocity": 0.0,
        "frustration_count": 0,
        "avg_score": 0.0,
        "irt_residual": 0.0,
        "tasks_spent": tasks_spent,
        "sample_size": 0,
    }


def should_publish_summary(tasks_spent: int, prev_mastery: float, curr_mastery: float) -> bool:
    """Плановый триггер: каждые 15 заданий ИЛИ скачок mastery ≥ 0.1."""
    return (tasks_spent > 0 and tasks_spent % 15 == 0) or (curr_mastery - prev_mastery >= 0.1)


def is_frustrated(frustration_count: int, velocity: float) -> bool:
    """OnFrustration: ≥ 2 ошибок подряд И velocity ≈ 0."""
    return frustration_count >= 2 and abs(velocity) < 0.05
