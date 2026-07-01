"""Вычисление MicroSummary из bandit_log для конкретного ученика и KC."""

from __future__ import annotations
import uuid

import sqlalchemy as sa

from shared.db import AsyncSessionLocal
from shared.config import retrieval as _cfg
from .selector import compute_p_correct

SUMMARY_WINDOW = _cfg.SUMMARY_WINDOW


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
      velocity            — динамика mastery_delta за последние N заданий
      frustration_count   — количество подряд идущих raw_score < 0.5
      avg_score           — средний raw_score
      hint_rate           — доля заданий с hints_used > 0
      irt_residual        — среднее |P_correct - raw_score|
      tasks_spent         — сколько заданий выдано по этой KC в текущем шаге плана
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT
                    bl.raw_score,
                    bl.hints_used,
                    bl.mastery_delta,
                    bl.irt_difficulty,
                    bl.context_vector
                FROM bandit_log bl
                WHERE bl.student_id = :student_id
                  AND bl.kc_id = :kc_id
                  AND bl.raw_score IS NOT NULL
                ORDER BY bl.recommended_at DESC
                LIMIT :window
            """),
            {"student_id": student_id, "kc_id": kc_id, "window": SUMMARY_WINDOW},
        )).fetchall()

    scores = [float(r[0]) for r in rows]

    if not scores:
        return _empty_summary(student_id, kc_id, mastery_current, tasks_spent)

    avg_score = sum(scores) / len(scores)
    mastery_deltas = [float(r[2]) for r in rows if r[2] is not None]

    # velocity: разница среднего mastery_delta между недавней и ранней половинами окна
    mid = len(mastery_deltas) // 2
    if mid > 0:
        early = sum(mastery_deltas[mid:]) / (len(mastery_deltas) - mid)
        late = sum(mastery_deltas[:mid]) / mid
        velocity = late - early
    elif mastery_deltas:
        velocity = sum(mastery_deltas) / len(mastery_deltas)
    else:
        velocity = 0.0

    # frustration: последовательные сырые провалы с начала окна (последние по времени)
    frustration_count = 0
    for score in scores:
        if score < 0.5:
            frustration_count += 1
        else:
            break

    hint_rate = sum(1 for r in rows if (r[1] or 0) > 0) / len(rows)

    residuals: list[float] = []
    for raw_score, _, _, irt_difficulty, context_vector in rows:
        if irt_difficulty is None:
            continue
        mastery_at_recommendation = mastery_current
        if context_vector and len(context_vector) > 0:
            mastery_at_recommendation = float(context_vector[0])
        predicted = compute_p_correct(mastery_at_recommendation, float(irt_difficulty))
        residuals.append(abs(predicted - float(raw_score)))
    irt_residual = sum(residuals) / len(residuals) if residuals else 0.0

    return {
        "student_id": str(student_id),
        "kc_id": kc_id,
        "mastery_current": mastery_current,
        "velocity": round(velocity, 4),
        "frustration_count": frustration_count,
        "avg_score": round(avg_score, 4),
        "hint_rate": round(hint_rate, 4),
        "irt_residual": round(irt_residual, 4),
        "tasks_spent": tasks_spent,
        "sample_size": len(scores),
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
        "hint_rate": 0.0,
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
