"""Logging helpers for macro estimator training labels."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa

from shared.db import AsyncSessionLocal
from .student_profile import MacroStudentProfile

logger = logging.getLogger(__name__)


async def log_macro_step_outcome(
    *,
    student_id: uuid.UUID,
    plan_id: uuid.UUID,
    kc_id: str,
    outcome_type: str,
    plan_step_id: uuid.UUID | None = None,
    tasks_spent: int | None = None,
    tasks_budget: int | None = None,
    difficulty_mode: str | None = None,
    mastery_current: float | None = None,
    recent_accuracy: float | None = None,
    velocity: float | None = None,
    reason: str | None = None,
    profile: MacroStudentProfile | None = None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                sa.text(
                    """
                    INSERT INTO macro_step_outcomes (
                        id, student_id, plan_id, plan_step_id, kc_id, outcome_type,
                        tasks_spent, tasks_budget, difficulty_mode, mastery_current,
                        recent_accuracy, velocity, reason, profile_snapshot, created_at
                    ) VALUES (
                        :id, :student_id, :plan_id, :plan_step_id, :kc_id, :outcome_type,
                        :tasks_spent, :tasks_budget, :difficulty_mode, :mastery_current,
                        :recent_accuracy, :velocity, :reason, :profile_snapshot, :created_at
                    )
                    """
                ),
                {
                    "id": uuid.uuid4(),
                    "student_id": student_id,
                    "plan_id": plan_id,
                    "plan_step_id": plan_step_id,
                    "kc_id": kc_id,
                    "outcome_type": outcome_type,
                    "tasks_spent": tasks_spent,
                    "tasks_budget": tasks_budget,
                    "difficulty_mode": difficulty_mode,
                    "mastery_current": mastery_current,
                    "recent_accuracy": recent_accuracy,
                    "velocity": velocity,
                    "reason": reason,
                    "profile_snapshot": json.dumps(profile.model_dump(mode="json")) if profile else None,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            await db.commit()
    except Exception:
        logger.warning(
            "Failed to log macro step outcome: student=%s plan=%s kc=%s outcome=%s",
            student_id,
            plan_id,
            kc_id,
            outcome_type,
        )
