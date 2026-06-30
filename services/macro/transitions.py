"""Offline RL transitions logger for plan lifecycle events.

Logs (state, action, reward, next_state, done) tuples to `macro_transitions`
table for future offline RL training (Decision Transformer, CQL, etc.).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa

from shared.db import AsyncSessionLocal

logger = logging.getLogger(__name__)


def _build_state(
    mastery_current: float,
    velocity: float,
    frustration_count: int,
    avg_score: float,
    tasks_spent: int,
    difficulty_mode: str | None = None,
) -> dict:
    return {
        "mastery": mastery_current,
        "velocity": velocity,
        "frustration_count": frustration_count,
        "avg_score": avg_score,
        "tasks_spent": tasks_spent,
        "difficulty_mode": difficulty_mode,
    }


async def log_transition(
    *,
    student_id: uuid.UUID,
    plan_id: uuid.UUID,
    kc_id: str,
    state: dict,
    action: str,
    action_payload: dict | None = None,
    reward: float | None = None,
    next_state: dict | None = None,
    done: bool = False,
    reason: str | None = None,
    diagnosis_reason: str | None = None,
    diagnosis_confidence: float | None = None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                sa.text("""
                    INSERT INTO macro_transitions
                        (id, student_id, plan_id, kc_id, state, action, action_payload,
                         reward, next_state, done, reason, diagnosis_reason,
                         diagnosis_confidence, created_at)
                    VALUES
                        (:id, :student_id, :plan_id, :kc_id, :state, :action, :action_payload,
                         :reward, :next_state, :done, :reason, :diagnosis_reason,
                         :diagnosis_confidence, :now)
                """),
                {
                    "id": uuid.uuid4(),
                    "student_id": student_id,
                    "plan_id": plan_id,
                    "kc_id": kc_id,
                    "state": json.dumps(state),
                    "action": action,
                    "action_payload": json.dumps(action_payload) if action_payload else None,
                    "reward": reward,
                    "next_state": json.dumps(next_state) if next_state else None,
                    "done": done,
                    "reason": reason,
                    "diagnosis_reason": diagnosis_reason,
                    "diagnosis_confidence": diagnosis_confidence,
                    "now": datetime.now(timezone.utc),
                },
            )
            await db.commit()
    except Exception:
        logger.warning(
            "Failed to log macro transition: student=%s action=%s kc=%s",
            student_id, action, kc_id,
        )
