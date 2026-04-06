"""
LearningSpeedModel — оценка скорости освоения KC учеником.

estimate_speed:
  Гибридная модель:
    - cluster_speed: средний прирост reward/задачу по кластеру для предмета
    - personal_speed: личная история из bandit_log ученика
    - α = min(1.0, n_personal / PERSONAL_HISTORY_FULL)
    - result = α × personal_speed + (1-α) × cluster_speed

Возвращает скорость как float [0..1]: средний Δmastery на одну задачу.
Если данных нет — возвращает DEFAULT_SPEED.
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa

from shared.db import AsyncSessionLocal

PERSONAL_HISTORY_FULL = 20   # при N личных записей α=1.0
DEFAULT_SPEED = 0.05          # дефолт: 5% прогресса на задачу
BANDIT_LOG_WINDOW = 30        # берём последние N записей из bandit_log


async def estimate_speed(
    kc_id: str,
    student_id: uuid.UUID,
    cluster_id: int | None,
) -> float:
    """
    Оценивает среднее Δmastery/задача для KC у данного ученика.

    Смешивает:
      1. cluster_speed — из cluster_task_stats (avg_reward как прокси mastery gain)
      2. personal_speed — из bandit_log ученика для этой KC

    Returns:
        float: оценка скорости обучения [0..1]
    """
    cluster_speed = await _cluster_speed(kc_id, cluster_id)
    personal_speed, n_personal = await _personal_speed(kc_id, student_id)

    alpha = min(1.0, n_personal / PERSONAL_HISTORY_FULL)
    if alpha == 0.0 or cluster_speed is None:
        return personal_speed if n_personal > 0 else DEFAULT_SPEED
    if cluster_speed is None:
        return personal_speed if n_personal > 0 else DEFAULT_SPEED

    return alpha * personal_speed + (1.0 - alpha) * cluster_speed


async def _cluster_speed(kc_id: str, cluster_id: int | None) -> float | None:
    """avg_reward из cluster_task_stats как прокси скорости обучения."""
    if cluster_id is None:
        return None
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("""
                SELECT AVG(avg_reward)
                FROM cluster_task_stats cts
                JOIN task_parts tp ON tp.task_id = cts.task_id
                WHERE cts.cluster_id = :cluster_id
                  AND tp.primary_kcs @> ARRAY[:kc_id]
                  AND cts.interaction_count > 0
            """),
            {"cluster_id": cluster_id, "kc_id": kc_id},
        )).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return None


async def _personal_speed(
    kc_id: str,
    student_id: uuid.UUID,
) -> tuple[float, int]:
    """
    Личная история из bandit_log.
    Считает средний reward по последним BANDIT_LOG_WINDOW записям для KC.

    Returns:
        (avg_reward, n_records)
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT reward
                FROM bandit_log
                WHERE student_id = :student_id
                  AND kc_id = :kc_id
                  AND reward IS NOT NULL
                ORDER BY recommended_at DESC
                LIMIT :window
            """),
            {"student_id": student_id, "kc_id": kc_id, "window": BANDIT_LOG_WINDOW},
        )).fetchall()

    if not rows:
        return 0.0, 0

    rewards = [float(r[0]) for r in rows]
    return sum(rewards) / len(rewards), len(rewards)


def blend_speed(
    personal_speed: float,
    n_personal: int,
    cluster_speed: float | None,
) -> float:
    """
    Чистая функция для тестирования: смешивает скорости без обращения к БД.
    """
    if cluster_speed is None:
        return personal_speed if n_personal > 0 else DEFAULT_SPEED
    alpha = min(1.0, n_personal / PERSONAL_HISTORY_FULL)
    return alpha * personal_speed + (1.0 - alpha) * cluster_speed
