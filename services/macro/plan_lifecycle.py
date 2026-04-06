"""
PlanLifecycleManager — создание и управление учебными планами.

create_plan:
  Режим 1 (target_mastery): BFS субграф → политика кластера → шаги плана
  Режим 2 (coverage): базовая структура без RL-политики (Phase 4)

evaluate_micro_summary:
  Три уровня эскалации:
    1. frustration / avg_score < threshold → difficulty_mode = consolidate
    2. velocity ≈ 0 > 3 задач → вставить слабый prereq
    3. velocity < 0 длительно → пересоздать план от текущего mastery

advance_step:
  Отметить текущий шаг completed, активировать следующий.

check_test_phase:
  Переключить difficulty_mode на "test" когда mastery близко к threshold.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import sqlalchemy as sa

from shared.db import AsyncSessionLocal

# Порог: если avg_score ниже этого — переключить на consolidate
FRUSTRATION_AVG_THRESHOLD = 0.45
# Порог: если velocity < 0 несколько раз подряд — пересоздать план
REPLAN_VELOCITY_THRESHOLD = -0.05
# test-фаза активируется когда mastery ≥ target - test_gap
TEST_GAP = 0.10


@dataclass
class PlanStep:
    kc_id: str
    difficulty_mode: str
    tasks_budget: int
    reason: str
    priority: float  # 1.0 = первый шаг, 0.9 = второй, ...


@dataclass
class PlanAction:
    action_type: Literal["set_difficulty_mode", "insert_prereq", "replan"]
    payload: dict


# ---------------------------------------------------------------------------
# create_plan
# ---------------------------------------------------------------------------

async def create_plan(
    student_id: uuid.UUID,
    mode: str,
    params: dict,
    steps: list[PlanStep],
) -> uuid.UUID:
    """
    Записывает план и шаги в БД. Возвращает plan_id.

    Args:
        student_id: UUID ученика
        mode: "target_mastery" | "coverage"
        params: {title, mastery_threshold, require_test, coverage_variant, task_budget}
        steps: упорядоченный список шагов плана
    """
    plan_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                INSERT INTO learning_plans
                    (id, student_id, title, created_at, goal_type,
                     mastery_threshold, require_test, coverage_variant, task_budget)
                VALUES
                    (:id, :student_id, :title, :created_at, :goal_type,
                     :mastery_threshold, :require_test, :coverage_variant, :task_budget)
            """),
            {
                "id": plan_id,
                "student_id": student_id,
                "title": params.get("title", f"Plan {plan_id}"),
                "created_at": now,
                "goal_type": mode,
                "mastery_threshold": params.get("mastery_threshold", 0.80),
                "require_test": params.get("require_test", False),
                "coverage_variant": params.get("coverage_variant"),
                "task_budget": params.get("task_budget"),
            },
        )

        for i, step in enumerate(steps):
            step_status = "in_progress" if i == 0 else "pending"
            await db.execute(
                sa.text("""
                    INSERT INTO plan_steps
                        (id, plan_id, kc_id, priority, status, reason,
                         inserted_at, difficulty_mode, tasks_budget, tasks_spent)
                    VALUES
                        (:id, :plan_id, :kc_id, :priority, :status, :reason,
                         :inserted_at, :difficulty_mode, :tasks_budget, 0)
                """),
                {
                    "id": uuid.uuid4(),
                    "plan_id": plan_id,
                    "kc_id": step.kc_id,
                    "priority": step.priority,
                    "status": step_status,
                    "reason": step.reason,
                    "inserted_at": now,
                    "difficulty_mode": step.difficulty_mode,
                    "tasks_budget": step.tasks_budget,
                },
            )

        await db.commit()

    return plan_id


# ---------------------------------------------------------------------------
# evaluate_micro_summary — три уровня эскалации
# ---------------------------------------------------------------------------

BUDGET_ALERT_RATIO = 0.20   # OnBudgetAlert если осталось < 20% бюджета


def evaluate_micro_summary(
    summary: dict,
    weakest_prereq_kc_id: str | None = None,
    tasks_budget: int | None = None,
) -> list[PlanAction]:
    """
    Анализирует MicroSummary и возвращает список действий для PlanLifecycleManager.

    Уровни эскалации (Mode 1):
      1. Слабый прогресс → consolidate mode
      2. Нулевая velocity несколько шагов → вставить prereq
      3. Отрицательная velocity → пересоздать план

    Coverage (Mode 2):
      OnBudgetAlert: если tasks_remaining < 20% бюджета

    Args:
        summary: dict из micro_summary.compute_micro_summary
        weakest_prereq_kc_id: KC пре-реквизита с наименьшим mastery (для уровня 2)
        tasks_budget: полный бюджет шага (для OnBudgetAlert)
    """
    actions: list[PlanAction] = []
    kc_id = summary["kc_id"]
    avg_score = summary.get("avg_score", 0.5)
    velocity = summary.get("velocity", 0.0)
    frustration_count = summary.get("frustration_count", 0)
    tasks_spent = summary.get("tasks_spent", 0)

    # OnBudgetAlert (Coverage Mode 2): мало задач осталось
    if tasks_budget is not None and tasks_budget > 0:
        tasks_remaining = tasks_budget - tasks_spent
        if tasks_remaining < tasks_budget * BUDGET_ALERT_RATIO:
            actions.append(PlanAction(
                action_type="set_difficulty_mode",
                payload={
                    "kc_id": kc_id,
                    "difficulty_mode": "test",
                    "reason": f"budget_alert: {tasks_remaining}/{tasks_budget} remaining",
                },
            ))
            return actions

    # Уровень 3: длительный регресс → переплан
    if velocity < REPLAN_VELOCITY_THRESHOLD and tasks_spent >= 10:
        actions.append(PlanAction(
            action_type="replan",
            payload={"reason": f"velocity={velocity:.3f} < {REPLAN_VELOCITY_THRESHOLD}, tasks_spent={tasks_spent}"},
        ))
        return actions  # эскалация 3 поглощает остальные

    # Уровень 2: нет прогресса + есть слабый пре-реквизит
    if (abs(velocity) < 0.02 and tasks_spent >= 5
            and weakest_prereq_kc_id is not None):
        actions.append(PlanAction(
            action_type="insert_prereq",
            payload={
                "kc_id": weakest_prereq_kc_id,
                "after_kc": kc_id,
                "reason": "velocity≈0, слабый пре-реквизит требует дополнительной работы",
            },
        ))

    # Уровень 1: фрустрация или низкий avg_score → consolidate
    if frustration_count >= 2 or avg_score < FRUSTRATION_AVG_THRESHOLD:
        actions.append(PlanAction(
            action_type="set_difficulty_mode",
            payload={"kc_id": kc_id, "difficulty_mode": "consolidate"},
        ))

    return actions


# ---------------------------------------------------------------------------
# Применение действий к БД
# ---------------------------------------------------------------------------

async def apply_plan_actions(
    student_id: uuid.UUID,
    actions: list[PlanAction],
    mastery: dict[str, float] | None = None,
) -> None:
    """Применяет PlanAction к базе данных."""
    if not actions:
        return

    async with AsyncSessionLocal() as db:
        for action in actions:
            if action.action_type == "set_difficulty_mode":
                await db.execute(
                    sa.text("""
                        UPDATE plan_steps
                           SET difficulty_mode = :mode
                         WHERE status = 'in_progress'
                           AND plan_id = (
                               SELECT id FROM learning_plans
                               WHERE student_id = :student_id
                               ORDER BY created_at DESC LIMIT 1
                           )
                    """),
                    {
                        "mode": action.payload["difficulty_mode"],
                        "student_id": student_id,
                    },
                )

            elif action.action_type == "insert_prereq":
                # Вставляем новый шаг только если KC ещё не в активном плане
                prereq_kc_id = action.payload["kc_id"]
                already = (await db.execute(
                    sa.text("""
                        SELECT 1 FROM plan_steps ps
                        JOIN learning_plans lp ON lp.id = ps.plan_id
                        WHERE lp.student_id = :student_id
                          AND ps.kc_id = :kc_id
                          AND ps.status IN ('in_progress', 'pending')
                          AND lp.id = (
                              SELECT id FROM learning_plans
                              WHERE student_id = :student_id
                              ORDER BY created_at DESC LIMIT 1
                          )
                        LIMIT 1
                    """),
                    {"student_id": student_id, "kc_id": prereq_kc_id},
                )).fetchone()
                if not already:
                    now = datetime.now(timezone.utc)
                    await db.execute(
                        sa.text("""
                            INSERT INTO plan_steps
                                (id, plan_id, kc_id, priority, status, reason,
                                 inserted_at, difficulty_mode, tasks_budget, tasks_spent)
                            VALUES (
                                :id,
                                (SELECT id FROM learning_plans
                                 WHERE student_id = :student_id
                                 ORDER BY created_at DESC LIMIT 1),
                                :kc_id, 0.95, 'pending', :reason,
                                :inserted_at, 'build', 15, 0
                            )
                        """),
                        {
                            "id": uuid.uuid4(),
                            "student_id": student_id,
                            "kc_id": prereq_kc_id,
                            "reason": action.payload["reason"],
                            "inserted_at": now,
                        },
                    )

        await db.commit()


# ---------------------------------------------------------------------------
# advance_step
# ---------------------------------------------------------------------------

async def advance_step(student_id: uuid.UUID) -> bool:
    """
    Завершает текущий in_progress шаг и активирует следующий pending шаг.
    Возвращает True если следующий шаг найден, False если план завершён.
    """
    async with AsyncSessionLocal() as db:
        # Завершаем текущий шаг
        await db.execute(
            sa.text("""
                UPDATE plan_steps
                   SET status = 'completed'
                 WHERE status = 'in_progress'
                   AND plan_id = (
                       SELECT id FROM learning_plans
                       WHERE student_id = :student_id
                       ORDER BY created_at DESC LIMIT 1
                   )
            """),
            {"student_id": student_id},
        )

        # Активируем следующий pending шаг (по убыванию priority)
        next_row = (await db.execute(
            sa.text("""
                UPDATE plan_steps
                   SET status = 'in_progress'
                 WHERE id = (
                     SELECT ps.id
                     FROM plan_steps ps
                     JOIN learning_plans lp ON lp.id = ps.plan_id
                     WHERE lp.student_id = :student_id
                       AND ps.status = 'pending'
                     ORDER BY ps.priority DESC
                     LIMIT 1
                 )
                RETURNING id
            """),
            {"student_id": student_id},
        )).fetchone()

        await db.commit()

    return next_row is not None


# ---------------------------------------------------------------------------
# check_and_advance — автоматический переход шага по mastery
# ---------------------------------------------------------------------------

ADVANCE_MIN_ACCURACY = 0.65  # recent_accuracy (avg_score за 20 задач) должна быть >= этого порога

async def check_and_advance(
    student_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    recent_accuracy: float = 1.0,
) -> bool:
    """
    Если mastery текущего шага >= mastery_threshold плана И recent_accuracy >= ADVANCE_MIN_ACCURACY
    → advance_step.

    recent_accuracy = avg_score за последние 20 задач (из MicroSummary).
    Защищает от случайного streak: требует устойчивой точности, не только хиты EMA.

    Ничего не делает если:
      - нет активного плана
      - активный шаг — другая KC
      - mastery ниже порога
      - recent_accuracy ниже ADVANCE_MIN_ACCURACY

    Возвращает True если шаг был продвинут.
    """
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("""
                SELECT lp.mastery_threshold, ps.kc_id
                FROM plan_steps ps
                JOIN learning_plans lp ON lp.id = ps.plan_id
                WHERE lp.student_id = :student_id
                  AND ps.status = 'in_progress'
                  AND lp.id = (
                      SELECT id FROM learning_plans
                      WHERE student_id = :student_id
                      ORDER BY created_at DESC LIMIT 1
                  )
                LIMIT 1
            """),
            {"student_id": student_id},
        )).fetchone()

    if not row:
        return False

    threshold, active_kc_id = row
    if active_kc_id != kc_id:
        return False
    if mastery_current < threshold:
        return False
    if recent_accuracy < ADVANCE_MIN_ACCURACY:
        return False

    return await advance_step(student_id)


# ---------------------------------------------------------------------------
# force_advance_if_budget_exceeded — принудительный переход при исчерпании бюджета
# ---------------------------------------------------------------------------

async def force_advance_if_budget_exceeded(
    student_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    tasks_spent: int,
) -> bool:
    """
    Если tasks_spent >= tasks_budget активного шага — принудительно завершаем шаг
    (partial) и переходим к следующему. Если следующего нет — создаём teacher_alert.

    Возвращает True если шаг был принудительно завершён.
    """
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("""
                SELECT ps.id, ps.tasks_budget, lp.id as plan_id
                FROM plan_steps ps
                JOIN learning_plans lp ON lp.id = ps.plan_id
                WHERE lp.student_id = :student_id
                  AND ps.status = 'in_progress'
                  AND ps.kc_id = :kc_id
                  AND lp.id = (
                      SELECT id FROM learning_plans
                      WHERE student_id = :student_id
                      ORDER BY created_at DESC LIMIT 1
                  )
                LIMIT 1
            """),
            {"student_id": student_id, "kc_id": kc_id},
        )).fetchone()

    if not row:
        return False

    step_id, tasks_budget, plan_id = row
    if tasks_budget is None or tasks_spent < tasks_budget:
        return False

    # Принудительно завершаем шаг как partial
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                UPDATE plan_steps
                   SET status = 'completed', reason = reason || ' [partial: budget exhausted]'
                 WHERE id = :step_id
            """),
            {"step_id": step_id},
        )
        # Пробуем активировать следующий pending шаг
        next_row = (await db.execute(
            sa.text("""
                UPDATE plan_steps
                   SET status = 'in_progress'
                 WHERE id = (
                     SELECT ps.id
                     FROM plan_steps ps
                     JOIN learning_plans lp ON lp.id = ps.plan_id
                     WHERE lp.student_id = :student_id
                       AND ps.status = 'pending'
                     ORDER BY ps.priority DESC
                     LIMIT 1
                 )
                RETURNING id
            """),
            {"student_id": student_id},
        )).fetchone()

        if not next_row:
            # Следующего шага нет — план исчерпан без достижения цели → teacher_alert
            await db.execute(
                sa.text("""
                    INSERT INTO teacher_alerts
                        (id, student_id, plan_id, kc_id, alert_type,
                         mastery_at_alert, tasks_spent, message, created_at)
                    VALUES
                        (:id, :student_id, :plan_id, :kc_id, 'plan_exhausted',
                         :mastery, :tasks_spent, :message, :now)
                """),
                {
                    "id": uuid.uuid4(),
                    "student_id": student_id,
                    "plan_id": plan_id,
                    "kc_id": kc_id,
                    "mastery": mastery_current,
                    "tasks_spent": tasks_spent,
                    "message": (
                        f"Ученик исчерпал бюджет всех шагов плана без достижения цели. "
                        f"Последний шаг: {kc_id}, mastery={mastery_current:.2f}, "
                        f"tasks_spent={tasks_spent}. Требуется вмешательство учителя."
                    ),
                    "now": datetime.now(timezone.utc),
                },
            )
        await db.commit()

    return True


async def create_plateau_alert(
    student_id: uuid.UUID,
    plan_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    tasks_spent: int,
) -> None:
    """Создаёт teacher_alert типа 'plateau' когда velocity≈0 длительное время."""
    async with AsyncSessionLocal() as db:
        # Не дублируем: если уже есть неразрешённый alert того же типа для этой KC — пропускаем
        existing = (await db.execute(
            sa.text("""
                SELECT id FROM teacher_alerts
                WHERE student_id = :student_id
                  AND kc_id = :kc_id
                  AND alert_type = 'plateau'
                  AND resolved_at IS NULL
            """),
            {"student_id": student_id, "kc_id": kc_id},
        )).fetchone()
        if existing:
            return

        await db.execute(
            sa.text("""
                INSERT INTO teacher_alerts
                    (id, student_id, plan_id, kc_id, alert_type,
                     mastery_at_alert, tasks_spent, message, created_at)
                VALUES
                    (:id, :student_id, :plan_id, :kc_id, 'plateau',
                     :mastery, :tasks_spent, :message, :now)
            """),
            {
                "id": uuid.uuid4(),
                "student_id": student_id,
                "plan_id": plan_id,
                "kc_id": kc_id,
                "mastery": mastery_current,
                "tasks_spent": tasks_spent,
                "message": (
                    f"Plateau: ученик не прогрессирует по {kc_id} "
                    f"уже {tasks_spent} заданий (mastery={mastery_current:.2f}). "
                    f"Возможно, нужна другая стратегия или помощь учителя."
                ),
                "now": datetime.now(timezone.utc),
            },
        )
        await db.commit()


# ---------------------------------------------------------------------------
# check_test_phase
# ---------------------------------------------------------------------------

def check_test_phase(mastery_current: float, target_mastery: float) -> bool:
    """
    Возвращает True если нужно переключить difficulty_mode на "test".
    Условие: ученик достаточно близко к цели (mastery >= target - TEST_GAP).
    """
    return mastery_current >= target_mastery - TEST_GAP - 1e-9
