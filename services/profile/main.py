"""Сервис профиля ученика."""

import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db import get_db

import math

from .bkt import update_mastery, compute_confidence
from .models import StudentModel, MasteryModel, InteractionModel, LearningPlanModel, PlanStepModel
from .repository import StudentRepository

app = FastAPI(title="Profile Service")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreateStudentRequest(BaseModel):
    grade: int
    student_id: Optional[uuid.UUID] = None
    review_mode: bool = False


class SubmitInteractionRequest(BaseModel):
    task_id: uuid.UUID
    part_id: str
    score: float                    # 0.0–1.0
    hints_used: int = 0
    time_spent_seconds: int = 0
    misconception_triggered: Optional[str] = None

    # KC затронутые этой частью задания
    primary_kcs: list[str]
    secondary_kcs: list[str] = []

    # Legacy BKT-совместимые параметры KC: online production path сейчас EMA-based
    p_transit: float = 0.1
    p_slip: float = 0.1
    p_guess: float = 0.2

    # Сложность задания по IRT — для surprise-бонуса в smooth_update
    irt_difficulty: Optional[float] = None

    # Период забывания KC в днях (передаётся из graph-сервиса)
    half_life_days: float = 30.0

    # Источник рекомендации — для аналитики
    recommendation_source: Optional[str] = None  # "zpd" | "exploration" | "placement" | "manual"


class UpdateReviewModeRequest(BaseModel):
    review_mode: bool


class ColdStartKC(BaseModel):
    kc_id: str
    grade_introduced: int
    max_prereq_grade: int = 0   # максимальный grade_introduced среди сильных пре-реквизитов


class ColdStartRequest(BaseModel):
    kcs: list[ColdStartKC]


class DiagnosticMasteryRequest(BaseModel):
    mastery: dict[str, float]


class MasteryResponse(BaseModel):
    kc_id: str
    probability: float
    probability_effective: float    # с учётом decay
    confidence: float
    last_practiced: datetime
    attempts_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_probability(record: MasteryModel, now: datetime, review_mode: bool = False) -> float:
    from .bkt import apply_decay
    if review_mode:
        return record.probability
    return apply_decay(record.probability, record.last_practiced, now)


def _model_to_mastery_response(record: MasteryModel, now: datetime) -> MasteryResponse:
    p_eff = _effective_probability(record, now)
    return MasteryResponse(
        kc_id=record.kc_id,
        probability=record.probability,
        probability_effective=p_eff,
        confidence=compute_confidence(
            record.attempts_count, record.recent_accuracy, p_eff, record.last_practiced, now,
        ),
        last_practiced=record.last_practiced,
        attempts_count=record.attempts_count,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/students", response_model=dict, status_code=201)
async def create_student(
    req: CreateStudentRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    student_id = req.student_id or uuid.uuid4()
    student = await repo.create(student_id=student_id, grade=req.grade, review_mode=req.review_mode)
    await db.commit()
    return {"student_id": str(student.student_id), "grade": student.grade, "review_mode": student.review_mode}


@app.get("/students/{student_id}")
async def get_student(
    student_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    student = await repo.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    now = datetime.utcnow()
    mastery = {}
    for r in student.mastery:
        p_eff = _effective_probability(r, now, student.review_mode)
        mastery[r.kc_id] = {
            "probability": r.probability,
            "probability_effective": p_eff,
            "last_practiced": r.last_practiced.isoformat(),
            "attempts_count": r.attempts_count,
            "consecutive_errors": r.consecutive_errors,
            "confidence": compute_confidence(
                r.attempts_count, r.recent_accuracy, p_eff, r.last_practiced, now,
            ),
        }
    return {
        "student_id": str(student.student_id),
        "grade": student.grade,
        "review_mode": student.review_mode,
        "created_at": student.created_at.isoformat(),
        "behavioral": {
            "guessing_rate": student.guessing_rate,
            "hint_dependence": student.hint_dependence,
        },
        "mastery": mastery,
    }


@app.get("/students/{student_id}/seen_tasks")
async def get_seen_tasks(
    student_id: uuid.UUID,
    ttl_days: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Возвращает task_id заданий которые студент уже видел.
    ttl_days: если задан — только задания за последние N дней.
    """
    repo = StudentRepository(db)
    task_ids = await repo.get_seen_tasks(student_id, ttl_days=ttl_days)
    return {"task_ids": [str(tid) for tid in task_ids]}


@app.patch("/students/{student_id}/review_mode", status_code=200)
async def set_review_mode(
    student_id: uuid.UUID,
    req: UpdateReviewModeRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    student = await repo.update_review_mode(student_id, req.review_mode)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    await db.commit()
    return {"student_id": str(student_id), "review_mode": student.review_mode}


@app.post("/students/{student_id}/mastery/cold_start", status_code=201)
async def cold_start_mastery(
    student_id: uuid.UUID,
    req: ColdStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Инициализирует mastery на основе класса ученика.
    Логика (prior знания нового студента):
      n-3 и ниже → 0.95  (давно пройдено, почти наверняка знает)
      n-2         → 0.90
      n-1         → 0.75  (прошлый год, в целом знает)
      n (текущий) → динамически:
        - все сильные пре-реквизиты из предыдущих классов (max_prereq_grade < n) → 0.50
        - есть сильный пре-реквизит из текущего класса (max_prereq_grade == n)   → 0.30
      выше n      → не инициализируем (студент ещё не проходил)
    """
    repo = StudentRepository(db)
    student = await repo.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    grade = student.grade
    now = datetime.utcnow()

    kc_probabilities: dict[str, float] = {}
    for kc in req.kcs:
        diff = grade - kc.grade_introduced
        if diff >= 3:
            kc_probabilities[kc.kc_id] = 0.95
        elif diff == 2:
            kc_probabilities[kc.kc_id] = 0.90
        elif diff == 1:
            kc_probabilities[kc.kc_id] = 0.75
        elif diff == 0:
            # Если KC текущего класса опирается на другую KC текущего класса —
            # студент с меньшей вероятностью её знает (0.30 vs 0.50)
            if kc.max_prereq_grade >= grade:
                kc_probabilities[kc.kc_id] = 0.30
            else:
                kc_probabilities[kc.kc_id] = 0.50
        # diff < 0: KC выше текущего класса — не инициализируем

    # Один bulk INSERT вместо N последовательных upsert
    count = await repo.bulk_initialize_mastery(student_id, kc_probabilities, now)

    await db.commit()
    return {"initialized_count": count, "mastery": kc_probabilities}


@app.post("/students/{student_id}/mastery/diagnostic", status_code=200)
async def apply_diagnostic_mastery(
    student_id: uuid.UUID,
    req: DiagnosticMasteryRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    student = await repo.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    now = datetime.utcnow()
    count = await repo.bulk_upsert_mastery(student_id, req.mastery, now)
    await db.commit()
    return {"updated_count": count, "mastery": req.mastery}


@app.get("/students/{student_id}/mastery")
async def get_all_mastery(
    student_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    records = await repo.get_all_mastery(student_id)
    now = datetime.utcnow()
    return [_model_to_mastery_response(r, now) for r in records]


@app.get("/students/{student_id}/mastery/{kc_id}")
async def get_mastery(
    student_id: uuid.UUID,
    kc_id: str,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)
    record = await repo.get_mastery(student_id, kc_id)
    if not record:
        raise HTTPException(status_code=404, detail="Mastery record not found")
    return _model_to_mastery_response(record, datetime.utcnow())


@app.post("/students/{student_id}/interactions", status_code=201)
async def submit_interaction(
    student_id: uuid.UUID,
    req: SubmitInteractionRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = StudentRepository(db)

    student = await repo.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    now = datetime.utcnow()

    # Сохранить лог взаимодействия
    interaction = InteractionModel(
        student_id=student_id,
        task_id=req.task_id,
        part_id=req.part_id,
        score=req.score,
        hints_used=req.hints_used,
        time_spent_seconds=req.time_spent_seconds,
        timestamp=now,
        misconception_triggered=req.misconception_triggered,
        recommendation_source=req.recommendation_source,
    )
    await repo.save_interaction(interaction)

    # Обновить mastery для каждого KC
    mastery_before: dict[str, float] = {}
    updated: dict[str, float] = {}
    consecutive_errors: dict[str, int] = {}

    kcs_with_roles = [
        (kc_id, "primary") for kc_id in req.primary_kcs
    ] + [
        (kc_id, "secondary") for kc_id in req.secondary_kcs
    ]

    for kc_id, role in kcs_with_roles:
        existing = await repo.get_mastery(student_id, kc_id)
        current_prob = existing.probability if existing else 0.0
        last_practiced = existing.last_practiced if existing else now
        prior_consecutive_errors = existing.consecutive_errors if existing else 0
        prior_consecutive_correct = existing.consecutive_correct if existing else 0

        # Точность за последние 5 задач по этой KC (из raw_score, не bandit reward)
        recent_rows = (await repo.db.execute(text(
            "SELECT raw_score FROM bandit_log"
            " WHERE student_id = :sid AND kc_id = :kc AND raw_score IS NOT NULL"
            " ORDER BY recommended_at DESC LIMIT 5"
        ), {"sid": student_id, "kc": kc_id})).fetchall()
        if recent_rows:
            recent_accuracy = sum(1 for r in recent_rows if float(r[0]) >= 0.5) / len(recent_rows)
        else:
            recent_accuracy = 0.0

        mastery_before[kc_id] = current_prob

        new_prob = update_mastery(
            current_probability=current_prob,
            last_practiced=last_practiced,
            score=req.score,
            hints_used=req.hints_used,
            p_transit=req.p_transit,
            p_slip=req.p_slip,
            p_guess=req.p_guess,
            kc_role=role,
            half_life_days=req.half_life_days,
            review_mode=student.review_mode,
            now=now,
            consecutive_errors=prior_consecutive_errors,
            consecutive_correct=prior_consecutive_correct,
            recent_accuracy=recent_accuracy,
            irt_difficulty=req.irt_difficulty,
            lr=student.estimated_lr,
        )

        record = await repo.upsert_mastery(
            student_id=student_id,
            kc_id=kc_id,
            probability=new_prob,
            now=now,
            p_transit=req.p_transit,
            p_slip=req.p_slip,
            p_guess=req.p_guess,
            score=req.score,
            recent_accuracy=recent_accuracy,
        )
        updated[kc_id] = new_prob
        consecutive_errors[kc_id] = record.consecutive_errors

    # Обновляем estimated_lr студента если известна сложность задания.
    # Residual = score - expected: если студент решил сложнее чем ожидалось →
    # он учится быстрее чем модель думает → увеличиваем estimated_lr.
    if req.irt_difficulty is not None and req.primary_kcs:
        residuals = []
        for kc_id in req.primary_kcs:
            m = max(0.01, min(0.99, mastery_before.get(kc_id, 0.05)))
            theta = math.log(m / (1.0 - m))
            expected = 1.0 / (1.0 + math.exp(-(theta - req.irt_difficulty)))
            residuals.append(req.score - expected)
        avg_residual = sum(residuals) / len(residuals)
        new_lr = student.estimated_lr + 0.005 * avg_residual
        student.estimated_lr = max(0.04, min(0.40, new_lr))

    # Обновляем behavioral signals (EMA, lr=0.1)
    _BEHAV_LR = 0.1
    # guessing_rate: правильный ответ без подсказок при низкой ожидаемой вероятности
    if req.irt_difficulty is not None and req.primary_kcs:
        avg_mastery = sum(mastery_before.get(kc, 0.05) for kc in req.primary_kcs) / len(req.primary_kcs)
        m = max(0.01, min(0.99, avg_mastery))
        theta = math.log(m / (1.0 - m))
        p_correct = 1.0 / (1.0 + math.exp(-(theta - req.irt_difficulty)))
        guess_signal = 1.0 if (req.score >= 0.8 and req.hints_used == 0 and p_correct < 0.3) else 0.0
        student.guessing_rate += _BEHAV_LR * (guess_signal - student.guessing_rate)
    # hint_dependence: EMA от использования подсказок
    hint_signal = 1.0 if req.hints_used > 0 else 0.0
    student.hint_dependence += _BEHAV_LR * (hint_signal - student.hint_dependence)

    await db.commit()
    return {
        "updated_mastery": updated,
        "mastery_before": mastery_before,
        "consecutive_errors": consecutive_errors,
    }


# ---------------------------------------------------------------------------
# Learning plan endpoints
# ---------------------------------------------------------------------------

class PlanStepRequest(BaseModel):
    kc_id: str
    priority: float          # 0..1, выше — срочнее
    reason: str = "goal"     # "goal" | "prerequisite" | "remedial"


class CreatePlanRequest(BaseModel):
    title: str
    steps: list[PlanStepRequest]


class CheckPlanRequest(BaseModel):
    updated_mastery: dict[str, float]
    consecutive_errors: dict[str, int] = {}


@app.post("/students/{student_id}/plan", status_code=201)
async def create_plan(
    student_id: uuid.UUID,
    req: CreatePlanRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Создаёт учебный план вручную. Заменяет предыдущий активный план.

    Возвращает zpd_warnings: список навыков из плана, которые недоступны прямо
    сейчас (mastery = 0), — учитель видит, что нужно сначала освоить предпосылки.
    """
    result = await db.execute(
        select(StudentModel).where(StudentModel.student_id == student_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Student not found")

    now = datetime.now(timezone.utc)
    plan = LearningPlanModel(
        id=uuid.uuid4(),
        student_id=student_id,
        title=req.title,
        created_at=now,
    )
    db.add(plan)
    await db.flush()

    for step_req in req.steps:
        step = PlanStepModel(
            id=uuid.uuid4(),
            plan_id=plan.id,
            kc_id=step_req.kc_id,
            priority=step_req.priority,
            status="pending",
            reason=step_req.reason,
            inserted_at=now,
        )
        db.add(step)

    # Проверяем достижимость: навыки с mastery=0 (нет записи) скорее всего
    # не входят в зону ближайшего развития — предпосылки ещё не освоены.
    zpd_warnings = []
    repo = StudentRepository(db)
    for step_req in req.steps:
        record = await repo.get_mastery(student_id, step_req.kc_id)
        if record is None or record.probability == 0.0:
            zpd_warnings.append(step_req.kc_id)

    await db.commit()
    return {
        "plan_id": str(plan.id),
        "steps_count": len(req.steps),
        "zpd_warnings": zpd_warnings,
        "zpd_warnings_message": (
            f"Следующие навыки пока недостижимы (нет освоенных предпосылок): "
            f"{', '.join(zpd_warnings)}"
            if zpd_warnings else None
        ),
    }


@app.get("/students/{student_id}/plan")
async def get_plan(
    student_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Возвращает последний план ученика с текущими статусами шагов."""
    result = await db.execute(
        select(LearningPlanModel)
        .where(LearningPlanModel.student_id == student_id)
        .options(selectinload(LearningPlanModel.steps))
        .order_by(LearningPlanModel.created_at.desc())
        .limit(1)
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=404, detail="No plan found for student")

    return {
        "plan_id": str(plan.id),
        "title": plan.title,
        "created_at": plan.created_at.isoformat(),
        "steps": [
            {
                "kc_id": s.kc_id,
                "priority": s.priority,
                "status": s.status,
                "reason": s.reason,
            }
            for s in plan.steps
        ],
    }


@app.post("/students/{student_id}/plan/check", status_code=200)
async def check_plan_thresholds(
    student_id: uuid.UUID,
    req: CheckPlanRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Проверяет пороги плана после BKT-обновления и обновляет статусы шагов.

    Правила:
      1. mastery > 0.9 → completed
      2. consecutive_errors >= 3 на completed KC → regression: in_progress + review шаг
    """
    result = await db.execute(
        select(LearningPlanModel)
        .where(LearningPlanModel.student_id == student_id)
        .options(selectinload(LearningPlanModel.steps))
        .order_by(LearningPlanModel.created_at.desc())
        .limit(1)
    )
    plan = result.scalar_one_or_none()
    if not plan:
        return {"changes": []}

    changes = []
    now = datetime.now(timezone.utc)

    for step in plan.steps:
        mastery = req.updated_mastery.get(step.kc_id)
        errors = req.consecutive_errors.get(step.kc_id, 0)

        if mastery is None:
            continue

        # Правило 1: освоено
        if mastery > 0.9 and step.status in ("pending", "in_progress"):
            step.status = "completed"
            changes.append({"kc_id": step.kc_id, "change": "completed"})

        # Правило 2: регрессия на завершённом KC
        elif errors >= 3 and step.status == "completed":
            step.status = "in_progress"
            # Добавить шаг повторения с высоким приоритетом
            review_step = PlanStepModel(
                id=uuid.uuid4(),
                plan_id=plan.id,
                kc_id=step.kc_id,
                priority=0.9,
                status="pending",
                reason="regression",
                inserted_at=now,
            )
            db.add(review_step)
            changes.append({"kc_id": step.kc_id, "change": "regression"})

        # Активируем pending шаг если предыдущий завершён
        elif step.status == "pending":
            completed_kc_ids = {s.kc_id for s in plan.steps if s.status == "completed"}
            # Простая эвристика: активируем pending с наивысшим приоритетом
            # если все предыдущие шаги (с более высоким приоритетом) завершены
            higher = [s for s in plan.steps if s.priority > step.priority and s.kc_id != step.kc_id]
            if not higher or all(s.kc_id in completed_kc_ids for s in higher):
                step.status = "in_progress"
                changes.append({"kc_id": step.kc_id, "change": "activated"})

    await db.commit()
    return {"changes": changes}


# ---------------------------------------------------------------------------
# A/B experiment endpoints
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "thompson_v1"


@app.post("/students/{student_id}/experiment", status_code=201)
async def assign_experiment(
    student_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Назначает ученика в контрольную или экспериментальную группу (50/50).
    Идемпотентен: повторный вызов возвращает уже назначенный вариант.
    """
    # Idempotency check
    existing = (await db.execute(
        text("SELECT variant FROM student_experiments WHERE student_id = :sid"),
        {"sid": student_id},
    )).fetchone()
    if existing:
        return {"experiment_id": EXPERIMENT_ID, "variant": existing[0]}

    variant = "treatment" if random.random() < 0.5 else "control"
    await db.execute(
        text("""
            INSERT INTO student_experiments (student_id, experiment_id, variant, assigned_at)
            VALUES (:student_id, :experiment_id, :variant, :assigned_at)
        """),
        {
            "student_id": student_id,
            "experiment_id": EXPERIMENT_ID,
            "variant": variant,
            "assigned_at": datetime.now(timezone.utc),
        },
    )
    await db.commit()
    return {"experiment_id": EXPERIMENT_ID, "variant": variant}


@app.get("/students/{student_id}/experiment")
async def get_experiment(
    student_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    row = (await db.execute(
        text("SELECT experiment_id, variant FROM student_experiments WHERE student_id = :sid"),
        {"sid": student_id},
    )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No experiment assignment found")
    return {"experiment_id": row[0], "variant": row[1]}
