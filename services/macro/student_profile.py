"""MacroStudentProfile model and persistence helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from pydantic import BaseModel

from shared.db import AsyncSessionLocal


class MacroStudentProfile(BaseModel):
    student_id: uuid.UUID
    version: int = 1
    updated_at: datetime
    target_kc_id: str | None = None

    uncertainty_level: float
    mastery_confidence_mean: float
    weak_prereq_fraction: float
    target_subgraph_mastery_mean: float

    learning_speed_global: float
    learning_speed_recent: float
    tasks_to_gain_01_mastery: float
    recovery_after_error: float

    frustration_risk: float
    stall_risk_baseline: float
    regression_risk_baseline: float

    pacing_mode: str
    budget_multiplier: float
    prereq_strictness: float
    test_readiness_bias: float
    step_granularity: float


PROFILE_COLUMNS = [
    "student_id",
    "version",
    "updated_at",
    "target_kc_id",
    "uncertainty_level",
    "mastery_confidence_mean",
    "weak_prereq_fraction",
    "target_subgraph_mastery_mean",
    "learning_speed_global",
    "learning_speed_recent",
    "tasks_to_gain_01_mastery",
    "recovery_after_error",
    "frustration_risk",
    "stall_risk_baseline",
    "regression_risk_baseline",
    "pacing_mode",
    "budget_multiplier",
    "prereq_strictness",
    "test_readiness_bias",
    "step_granularity",
]


def _row_to_profile(row) -> MacroStudentProfile:
    return MacroStudentProfile(**{column: row[idx] for idx, column in enumerate(PROFILE_COLUMNS)})


def _profile_params(profile: MacroStudentProfile) -> dict:
    params = profile.model_dump()
    params["updated_at"] = params["updated_at"] or datetime.now(timezone.utc)
    return params


async def get_macro_student_profile(student_id: uuid.UUID) -> MacroStudentProfile | None:
    async with AsyncSessionLocal() as db:
        row = (
            await db.execute(
                sa.text(
                    f"SELECT {', '.join(PROFILE_COLUMNS)} "
                    "FROM macro_student_profiles WHERE student_id = :student_id"
                ),
                {"student_id": student_id},
            )
        ).fetchone()
    if not row:
        return None
    return _row_to_profile(row)


async def save_macro_student_profile(
    profile: MacroStudentProfile,
    *,
    create_snapshot: bool = True,
) -> MacroStudentProfile:
    params = _profile_params(profile)

    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text(
                """
                INSERT INTO macro_student_profiles (
                    student_id, version, updated_at, target_kc_id,
                    uncertainty_level, mastery_confidence_mean,
                    weak_prereq_fraction, target_subgraph_mastery_mean,
                    learning_speed_global, learning_speed_recent,
                    tasks_to_gain_01_mastery, recovery_after_error,
                    frustration_risk, stall_risk_baseline, regression_risk_baseline,
                    pacing_mode, budget_multiplier, prereq_strictness,
                    test_readiness_bias, step_granularity
                ) VALUES (
                    :student_id, :version, :updated_at, :target_kc_id,
                    :uncertainty_level, :mastery_confidence_mean,
                    :weak_prereq_fraction, :target_subgraph_mastery_mean,
                    :learning_speed_global, :learning_speed_recent,
                    :tasks_to_gain_01_mastery, :recovery_after_error,
                    :frustration_risk, :stall_risk_baseline, :regression_risk_baseline,
                    :pacing_mode, :budget_multiplier, :prereq_strictness,
                    :test_readiness_bias, :step_granularity
                )
                ON CONFLICT (student_id) DO UPDATE SET
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at,
                    target_kc_id = EXCLUDED.target_kc_id,
                    uncertainty_level = EXCLUDED.uncertainty_level,
                    mastery_confidence_mean = EXCLUDED.mastery_confidence_mean,
                    weak_prereq_fraction = EXCLUDED.weak_prereq_fraction,
                    target_subgraph_mastery_mean = EXCLUDED.target_subgraph_mastery_mean,
                    learning_speed_global = EXCLUDED.learning_speed_global,
                    learning_speed_recent = EXCLUDED.learning_speed_recent,
                    tasks_to_gain_01_mastery = EXCLUDED.tasks_to_gain_01_mastery,
                    recovery_after_error = EXCLUDED.recovery_after_error,
                    frustration_risk = EXCLUDED.frustration_risk,
                    stall_risk_baseline = EXCLUDED.stall_risk_baseline,
                    regression_risk_baseline = EXCLUDED.regression_risk_baseline,
                    pacing_mode = EXCLUDED.pacing_mode,
                    budget_multiplier = EXCLUDED.budget_multiplier,
                    prereq_strictness = EXCLUDED.prereq_strictness,
                    test_readiness_bias = EXCLUDED.test_readiness_bias,
                    step_granularity = EXCLUDED.step_granularity
                """
            ),
            params,
        )

        if create_snapshot:
            snapshot_params = dict(params)
            snapshot_params["id"] = uuid.uuid4()
            snapshot_params["snapshot_created_at"] = datetime.now(timezone.utc)
            await db.execute(
                sa.text(
                    """
                    INSERT INTO macro_student_profile_snapshots (
                        id, student_id, version, updated_at, target_kc_id,
                        uncertainty_level, mastery_confidence_mean,
                        weak_prereq_fraction, target_subgraph_mastery_mean,
                        learning_speed_global, learning_speed_recent,
                        tasks_to_gain_01_mastery, recovery_after_error,
                        frustration_risk, stall_risk_baseline, regression_risk_baseline,
                        pacing_mode, budget_multiplier, prereq_strictness,
                        test_readiness_bias, step_granularity, snapshot_created_at
                    ) VALUES (
                        :id, :student_id, :version, :updated_at, :target_kc_id,
                        :uncertainty_level, :mastery_confidence_mean,
                        :weak_prereq_fraction, :target_subgraph_mastery_mean,
                        :learning_speed_global, :learning_speed_recent,
                        :tasks_to_gain_01_mastery, :recovery_after_error,
                        :frustration_risk, :stall_risk_baseline, :regression_risk_baseline,
                        :pacing_mode, :budget_multiplier, :prereq_strictness,
                        :test_readiness_bias, :step_granularity, :snapshot_created_at
                    )
                    """
                ),
                snapshot_params,
            )

        await db.commit()

    return profile
