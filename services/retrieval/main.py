"""Сервис подбора задания (Retrieval).

Горячий путь:
  1. Получить mastery + seen_tasks из Profile
  2. Получить ZPD-кандидатов из Graph
  3. Выбрать KC (subject rotation)
  4. Получить задания для KC из TaskBank (с exclude фильтром)
  5. Выбрать задание — Thompson Sampling с контекстным вектором
  6. Записать в bandit_log (context_vector, reward=NULL)
  7. Вернуть задание + метаданные рекомендации

После ответа (PATCH /bandit_log/reward):
  8. Обновить reward в bandit_log
  9. Обновить Thompson Sampling модель (A, b) в bandit_model
"""

import math
import os
import random
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.db import AsyncSessionLocal
from . import clients
from shared.config import retrieval as _cfg
from .thompson import ThompsonModel, GLOBAL_CLUSTER_ID
from .selector import ZPDEntry, select_kc_from_zpd, compute_zpd_target_difficulty, compute_p_correct, filter_tasks_by_irt
from .diagnostic_cat import CATState, select_diagnostic_kc, select_diagnostic_task, update_cat_state
from .micro_summary import compute_micro_summary, is_frustrated
from .kafka_producer import publish_micro_summary
from services.clustering.cluster import (
    assign_cluster_for_new_student,
    save_student_cluster,
    ensure_centroids_loaded,
    should_reassign_cluster,
)

CONTEXT_DIM = _cfg.CONTEXT_DIM
CLUSTER_EXPLORE_RATE = _cfg.CLUSTER_EXPLORE_RATE
CLUSTER_EXPLORE_THRESHOLD = _cfg.CLUSTER_EXPLORE_THRESHOLD
EPSILON_GREEDY_RATE = _cfg.EPSILON_GREEDY_RATE
KC_COOLDOWN_WINDOW = _cfg.KC_COOLDOWN_WINDOW
KC_COOLDOWN_MAX = _cfg.KC_COOLDOWN_MAX
PHASE1_TASK_THRESHOLD = _cfg.PHASE1_TASK_THRESHOLD
TARGET_ZPD_ACCURACY = _cfg.TARGET_ZPD_ACCURACY
CLUSTER_REASSIGN_INTERVAL = _cfg.CLUSTER_REASSIGN_INTERVAL
CLUSTER_REASSIGN_MIN_TASKS = _cfg.CLUSTER_REASSIGN_MIN_TASKS

MODE_PARAMS = {
    "build":       {"cluster_explore": 0.20, "epsilon": 0.05, "irt_floor": 0.20, "irt_ceiling": 0.90},
    "consolidate": {"cluster_explore": 0.10, "epsilon": 0.10, "irt_floor": 0.40, "irt_ceiling": 0.95},
    "test":        {"cluster_explore": 0.00, "epsilon": 0.00, "irt_floor": 0.15, "irt_ceiling": 0.85},
    "diagnostic":  {"cluster_explore": 0.00, "epsilon": 0.00, "irt_floor": 0.30, "irt_ceiling": 0.70},
}

_rng = random.Random()

app = FastAPI(title="Retrieval Service")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    student_id: uuid.UUID
    last_subject: Optional[str] = None   # предмет последнего задания — для rotation


class RecommendResponse(BaseModel):
    task_id: str
    kc_id: str
    subject: str
    recommendation_source: str           # "thompson_sampling"
    half_life_days: float                # передаётся в profile при submit
    task: dict                           # полные данные задания


class UpdateRewardRequest(BaseModel):
    student_id: uuid.UUID
    task_id: uuid.UUID
    reward: float
    raw_score: float | None = None
    hints_used: int | None = None
    time_spent_seconds: int | None = None
    irt_difficulty: float | None = None
    mastery_delta: float | None = None
    answered_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTROL_GROUP_ENABLED = os.getenv("ENABLE_CONTROL_GROUP", "false").lower() == "true"


async def _get_experiment_variant(student_id: uuid.UUID) -> str:
    """Возвращает 'treatment' | 'control'.

    Если ENABLE_CONTROL_GROUP != 'true' — всегда 'treatment' (контрол отключён).
    Включать только для явных A/B симуляций: ENABLE_CONTROL_GROUP=true.
    """
    if not _CONTROL_GROUP_ENABLED:
        return "treatment"
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text(
                "SELECT variant FROM student_experiments WHERE student_id = :sid"
            ),
            {"sid": student_id},
        )).fetchone()
    return row[0] if row else "treatment"


async def _get_plan_info(
    student_id: uuid.UUID,
) -> tuple[dict[str, float], str, str | None, str | None, uuid.UUID | None]:
    """
    Возвращает:
      - {kc_id: priority} для активных шагов плана
      - difficulty_mode активного шага (in_progress) или 'build' если нет
      - next_plan_kc_id — kc_id следующего pending шага (для bridge bonus) или None
      - active_plan_kc_id — kc_id текущего in_progress шага или None (для hard lock)
      - active_plan_step_id — UUID активного шага или None
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT ps.kc_id, ps.priority, ps.status, ps.difficulty_mode, ps.id
                FROM plan_steps ps
                JOIN learning_plans lp ON lp.id = ps.plan_id
                WHERE lp.student_id = :student_id
                  AND ps.status IN ('pending', 'in_progress')
                  AND lp.id = (
                      SELECT id FROM learning_plans
                      WHERE student_id = :student_id
                      ORDER BY created_at DESC LIMIT 1
                  )
                ORDER BY ps.priority DESC
            """),
            {"student_id": student_id},
        )).fetchall()

    priorities: dict[str, float] = {}
    difficulty_mode = "build"
    next_plan_kc_id: str | None = None
    active_plan_kc_id: str | None = None
    active_plan_step_id: uuid.UUID | None = None

    in_progress_seen = False
    for r in rows:
        kc_id, priority, status, dm, step_id = r[0], float(r[1]), r[2], r[3], r[4]
        priorities[kc_id] = priority
        if status == "in_progress" and not in_progress_seen:
            difficulty_mode = dm or "build"
            active_plan_kc_id = kc_id
            active_plan_step_id = step_id
            in_progress_seen = True
        elif status == "pending" and in_progress_seen and next_plan_kc_id is None:
            next_plan_kc_id = kc_id

    return priorities, difficulty_mode, next_plan_kc_id, active_plan_kc_id, active_plan_step_id


async def _get_cluster_id(student_id: uuid.UUID) -> int | None:
    """Возвращает cluster_id ученика или None если ещё не назначен."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("SELECT cluster_id FROM student_clusters WHERE student_id = :sid"),
            {"sid": student_id},
        )).fetchone()
    return int(row[0]) if row else None


async def _maybe_refresh_cluster_assignment(
    student_id: uuid.UUID,
    mastery: dict[str, float],
    confidence: dict[str, float],
    task_count: int,
    current_cluster_id: int | None,
) -> int | None:
    if not should_reassign_cluster(
        task_count,
        min_tasks=CLUSTER_REASSIGN_MIN_TASKS,
        interval=CLUSTER_REASSIGN_INTERVAL,
    ):
        return current_cluster_id
    new_cluster = assign_cluster_for_new_student(mastery, confidence)
    if new_cluster is None:
        return current_cluster_id
    if current_cluster_id != new_cluster:
        await save_student_cluster(
            student_id,
            new_cluster,
            reason="periodic_refresh",
            task_count=task_count,
        )
        logger.info(
            "cluster reassigned student=%s old=%s new=%s task_count=%d",
            student_id, current_cluster_id, new_cluster, task_count,
        )
    return new_cluster


async def _get_cluster_task_stats(
    cluster_id: int,
    task_ids: list[str],
) -> dict[str, tuple[float, int]]:
    """
    Возвращает {task_id: (avg_reward, interaction_count)} из cluster_task_stats.
    Задания без статистики получают (0.0, 0).
    """
    if not task_ids:
        return {}
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text(
                "SELECT task_id::text, avg_reward, interaction_count"
                " FROM cluster_task_stats"
                " WHERE cluster_id = :cid AND task_id = ANY(:task_ids)"
            ),
            {"cid": cluster_id, "task_ids": task_ids},
        )).fetchall()
    return {r[0]: (float(r[1]), int(r[2])) for r in rows}


def _build_context(
    mastery_kc: float,
    grade: int,
    avg_reward: float,
    count: int,
    errors_streak: int = 0,
    prereq_masteries: list[float] | None = None,
    task_type: str = "procedural",
    n_steps: int = 1,
) -> np.ndarray:
    """
    Строит 13-мерный контекстный вектор.
    x[0]   = mastery текущей KC
    x[1..4] = mastery пре-реквизитов (по убыванию силы связи, паддинг нулями)
    x[5]   = consecutive_errors
    x[6]   = grade / 11.0
    x[7]   = avg_reward из cluster_task_stats
    x[8]   = log(interaction_count + 1)
    x[9]   = is_conceptual (0/1)
    x[10]  = is_word_problem (0/1)
    x[11]  = is_mixed (0/1)
    x[12]  = n_steps / 5.0
    """
    x = np.zeros(CONTEXT_DIM, dtype=np.float64)
    x[0] = mastery_kc
    for i, m in enumerate((prereq_masteries or [])[:4]):
        x[i + 1] = m
    x[5] = float(errors_streak)
    x[6] = grade / 11.0
    x[7] = avg_reward
    x[8] = math.log(count + 1)
    x[9]  = 1.0 if task_type == "conceptual"   else 0.0
    x[10] = 1.0 if task_type == "word_problem" else 0.0
    x[11] = 1.0 if task_type == "mixed"        else 0.0
    x[12] = min(n_steps, 5) / 5.0
    return x


async def _get_cat_state(student_id: uuid.UUID) -> tuple[CATState | None, bool]:
    """Returns (CATState, completed). None if no CAT session exists yet."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text(
                "SELECT kc_theta, kc_n, tasks_used, completed"
                " FROM diagnostic_cat_state WHERE student_id = :sid"
            ),
            {"sid": student_id},
        )).fetchone()
    if not row:
        return None, False
    if row[3]:
        return None, True
    state = CATState(kc_theta=row[0], kc_n=row[1], tasks_used=row[2])
    return state, False


async def _save_cat_state(student_id: uuid.UUID, state: CATState, completed: bool = False) -> None:
    import json
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                INSERT INTO diagnostic_cat_state
                    (student_id, kc_theta, kc_n, tasks_used, completed, created_at, completed_at)
                VALUES (:sid, :theta, :kc_n, :tasks_used, :completed, :now, :completed_at)
                ON CONFLICT (student_id) DO UPDATE
                    SET kc_theta = EXCLUDED.kc_theta,
                        kc_n = EXCLUDED.kc_n,
                        tasks_used = EXCLUDED.tasks_used,
                        completed = EXCLUDED.completed,
                        completed_at = EXCLUDED.completed_at
            """),
            {
                "sid": student_id,
                "theta": json.dumps(state.kc_theta),
                "kc_n": json.dumps(state.kc_n),
                "tasks_used": state.tasks_used,
                "completed": completed,
                "now": now,
                "completed_at": now if completed else None,
            },
        )
        await db.commit()


async def _get_student_task_count(student_id: uuid.UUID) -> int:
    """Количество заданий, на которые студент уже дал ответ."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text(
                "SELECT COUNT(*) FROM bandit_log"
                " WHERE student_id = :sid AND reward IS NOT NULL"
            ),
            {"sid": student_id},
        )).fetchone()
    return int(row[0]) if row else 0


async def _get_recent_kcs(student_id: uuid.UUID, window: int = KC_COOLDOWN_WINDOW) -> list[str]:
    """Возвращает kc_id последних N рекомендаций ученика (для cooldown)."""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text(
                "SELECT kc_id FROM bandit_log "
                "WHERE student_id = :sid "
                "ORDER BY recommended_at DESC LIMIT :n"
            ),
            {"sid": student_id, "n": window},
        )).fetchall()
    return [r[0] for r in rows]


async def _load_student_model(
    db,
    student_id: uuid.UUID,
    kc_id: str,
    fallback_cluster_id: int,
) -> ThompsonModel:
    """
    Загружает личную Thompson Sampling модель студента.
    При первом обращении инициализируется из кластерной модели (transfer learning).
    Storage: A precision matrix + b reward vector.
    """
    row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM student_bandit_model"
            " WHERE student_id = :sid AND kc_id = :kc_id FOR UPDATE"
        ),
        {"sid": student_id, "kc_id": kc_id},
    )).fetchone()
    if row:
        return ThompsonModel.from_bytes(kc_id, fallback_cluster_id, bytes(row[0]), bytes(row[1]))

    cluster_row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM bandit_model"
            " WHERE cluster_id = :cluster_id AND kc_id = :kc_id"
        ),
        {"cluster_id": fallback_cluster_id, "kc_id": kc_id},
    )).fetchone()
    if cluster_row:
        return ThompsonModel.from_bytes(kc_id, fallback_cluster_id, bytes(cluster_row[0]), bytes(cluster_row[1]))
    return ThompsonModel.init(kc_id, fallback_cluster_id)


async def _save_student_model(db, student_id: uuid.UUID, model: ThompsonModel) -> None:
    a_bytes, b_bytes = model.to_bytes()
    await db.execute(
        sa.text("""
            INSERT INTO student_bandit_model (student_id, kc_id, a_matrix, b_vector, updated_at)
            VALUES (:sid, :kc_id, :a_matrix, :b_vector, :updated_at)
            ON CONFLICT (student_id, kc_id)
            DO UPDATE SET a_matrix = EXCLUDED.a_matrix,
                          b_vector = EXCLUDED.b_vector,
                          updated_at = EXCLUDED.updated_at
        """),
        {
            "sid": student_id,
            "kc_id": model.kc_id,
            "a_matrix": a_bytes,
            "b_vector": b_bytes,
            "updated_at": datetime.now(timezone.utc),
        },
    )


async def _load_model(db, kc_id: str, cluster_id: int) -> ThompsonModel:
    """Загружает модель кластера из БД. Создаёт новую если не существует."""
    row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM bandit_model"
            " WHERE cluster_id = :cluster_id AND kc_id = :kc_id FOR UPDATE"
        ),
        {"cluster_id": cluster_id, "kc_id": kc_id},
    )).fetchone()
    if row:
        return ThompsonModel.from_bytes(kc_id, cluster_id, bytes(row[0]), bytes(row[1]))
    return ThompsonModel.init(kc_id, cluster_id)


async def _save_model(db, model: ThompsonModel) -> None:
    a_bytes, b_bytes = model.to_bytes()
    await db.execute(
        sa.text("""
            INSERT INTO bandit_model (cluster_id, kc_id, a_matrix, b_vector, updated_at)
            VALUES (:cluster_id, :kc_id, :a_matrix, :b_vector, :updated_at)
            ON CONFLICT (cluster_id, kc_id)
            DO UPDATE SET a_matrix = EXCLUDED.a_matrix,
                          b_vector = EXCLUDED.b_vector,
                          updated_at = EXCLUDED.updated_at
        """),
        {
            "cluster_id": model.cluster_id,
            "kc_id": model.kc_id,
            "a_matrix": a_bytes,
            "b_vector": b_bytes,
            "updated_at": datetime.now(timezone.utc),
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Подбирает следующее задание для ученика через Thompson Sampling.
    Проходит по ZPD-кандидатам до первого KC у которого есть задания в TaskBank.
    """
    started = time.perf_counter()
    async with httpx.AsyncClient(trust_env=False) as http:

        # 1. Профиль ученика
        try:
            student = await clients.get_student(http, req.student_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Student not found")
            raise HTTPException(status_code=502, detail="Profile service error")

        grade = student["grade"]
        mastery_raw = student.get("mastery", {})
        mastery: dict[str, float] = {
            kc_id: v["probability_effective"]
            for kc_id, v in mastery_raw.items()
        }
        mastery_confidence: dict[str, float] = {
            kc_id: float(v.get("confidence", 0.0))
            for kc_id, v in mastery_raw.items()
        }
        consecutive_errors: dict[str, int] = {
            kc_id: v.get("consecutive_errors", 0)
            for kc_id, v in mastery_raw.items()
        }

        # 2. Seen tasks (только за последние 90 дней — задания старше TTL повторяются)
        try:
            seen_tasks = await clients.get_seen_tasks(http, req.student_id, ttl_days=90)
        except httpx.HTTPStatusError as e:
            logger.warning("seen_tasks fetch failed for student=%s: %s", req.student_id, e)
            seen_tasks = []

        # 3. ZPD кандидаты
        try:
            zpd_raw = await clients.get_zpd(http, mastery, grade)
        except httpx.HTTPStatusError:
            raise HTTPException(status_code=502, detail="Graph service error")

        if not zpd_raw:
            logger.info("recommendation skipped: empty ZPD for student=%s", req.student_id)
            raise HTTPException(status_code=404, detail="ZPD пуст — нет подходящих тем")

        # Приоритеты из учебного плана + difficulty_mode + next_plan_kc + текущий шаг
        plan_priorities, difficulty_mode, next_plan_kc_id, active_plan_kc_id, active_plan_step_id = await _get_plan_info(req.student_id)

        candidates = [
            ZPDEntry(
                kc_id=c["kc_id"],
                subject=c.get("subject", ""),
                difficulty_base=c["difficulty_base"],
                mastery_effective=c["mastery_effective"],
                ready=c["ready"],
                plan_priority=plan_priorities.get(c["kc_id"], 0.0),
            )
            for c in zpd_raw
        ]

        # 4. Выбор KC:
        # Если план активен — hard lock на текущий шаг плана.
        # Если KC шага не в ZPD (например, count-вариант выбрал KC близкую к порогу),
        # добавляем её синтетически — план важнее ZPD-фильтра.
        # Fallback на ZPD только если для plan KC нет заданий в task_bank.
        selection_reason = "zpd_rotation"
        if active_plan_kc_id:
            selection_reason = "plan_lock"
            plan_step_candidates = [c for c in candidates if c.kc_id == active_plan_kc_id]
            if not plan_step_candidates:
                # KC не в ZPD — добавляем синтетически с текущим mastery
                plan_m = mastery.get(active_plan_kc_id, 0.0)
                plan_step_candidates = [ZPDEntry(
                    kc_id=active_plan_kc_id,
                    subject="",
                    difficulty_base=plan_m,
                    mastery_effective=plan_m,
                    ready=True,
                    plan_priority=1.0,
                )]
            ordered = plan_step_candidates + [c for c in candidates if c.kc_id != active_plan_kc_id]
        else:
            preferred = select_kc_from_zpd(candidates, req.last_subject)
            if preferred and req.last_subject and preferred.subject != req.last_subject:
                selection_reason = "subject_rotation"
            ordered = [preferred] + [c for c in candidates if c.kc_id != preferred.kc_id] if preferred else candidates

        # 5. A/B вариант — определяем алгоритм выбора задания
        variant = await _get_experiment_variant(req.student_id)

        # 5a. Diagnostic CAT for cold start
        cat_state, cat_completed = await _get_cat_state(req.student_id)
        if cat_state is None and not cat_completed:
            cat_state = CATState.from_mastery(mastery)

        # 5b. Фаза студента и кластер
        task_count = await _get_student_task_count(req.student_id)
        in_cat = cat_state is not None and not cat_state.is_complete and not cat_completed
        phase1 = not in_cat and task_count < PHASE1_TASK_THRESHOLD

        cluster_id = await _get_cluster_id(req.student_id) if variant == "treatment" else None

        # Граница фаз 1→2: переназначаем кластер по реальному поведению
        if variant == "treatment" and task_count == PHASE1_TASK_THRESHOLD:
            new_cluster = assign_cluster_for_new_student(mastery, mastery_confidence)
            if new_cluster is not None:
                await save_student_cluster(
                    req.student_id,
                    new_cluster,
                    reason="phase_boundary",
                    task_count=task_count,
                )
                cluster_id = new_cluster
        elif variant == "treatment":
            cluster_id = await _maybe_refresh_cluster_assignment(
                req.student_id,
                mastery,
                mastery_confidence,
                task_count,
                cluster_id,
            )

        # 5c. Cooldown: последние KC этого ученика
        recent_kcs = await _get_recent_kcs(req.student_id)

        selected_task = None
        selected_kc: ZPDEntry | None = None
        selected_context: np.ndarray | None = None
        selected_source: str = "thompson_sampling"
        plan_fallback_occurred = False
        selected_irt_fallback = False

        for kc_entry in ordered:
            # ── Cooldown check ──────────────────────────────────────────────
            # Если KC мелькала слишком часто в последних рекомендациях и есть
            # альтернативы — пропускаем. Решает проблему зависания kc_point_line_plane.
            if (len(ordered) > 1
                    and recent_kcs.count(kc_entry.kc_id) >= KC_COOLDOWN_MAX):
                continue

            try:
                tasks = await clients.get_tasks_by_kc(
                    http,
                    kc_id=kc_entry.kc_id,
                    grade_min=grade,
                    exclude_task_ids=seen_tasks,
                )
            except httpx.HTTPStatusError:
                continue

            if not tasks:
                if active_plan_kc_id and kc_entry.kc_id == active_plan_kc_id:
                    plan_fallback_occurred = True
                    logger.warning(
                        "plan fallback: no tasks for plan KC=%s student=%s",
                        active_plan_kc_id, req.student_id,
                    )
                continue

            # IRT pre-filter: убираем задания по mode-aware boundaries
            mode_p = MODE_PARAMS.get(difficulty_mode, MODE_PARAMS["build"])
            tasks, irt_fallback_used = filter_tasks_by_irt(
                tasks, kc_entry.mastery_effective,
                floor=mode_p["irt_floor"], ceiling=mode_p["irt_ceiling"],
            )
            if irt_fallback_used:
                logger.info(
                    "irt fallback used student=%s kc=%s mode=%s candidate_count=%d",
                    req.student_id, kc_entry.kc_id, difficulty_mode, len(tasks),
                )

            # Пре-реквизиты KC для контекстного вектора (x[1..4])
            try:
                prereqs = await clients.get_kc_prerequisites(http, kc_entry.kc_id)
                prereq_masteries = [mastery.get(p["kc_id"], 0.0) for p in prereqs]
            except httpx.HTTPStatusError:
                prereq_masteries = []

            # 5d. Diagnostic CAT: select maximally informative task
            if in_cat and cat_state is not None:
                diag_kc = select_diagnostic_kc(cat_state, [c.kc_id for c in candidates])
                if diag_kc and diag_kc == kc_entry.kc_id:
                    theta = cat_state.kc_theta.get(kc_entry.kc_id, 0.0)
                    chosen = select_diagnostic_task(tasks, theta)
                    if chosen:
                        _p = (chosen.get("parts") or [{}])[0]
                        ctx = _build_context(
                            mastery_kc=kc_entry.mastery_effective,
                            grade=grade,
                            avg_reward=0.0,
                            count=0,
                            errors_streak=0,
                            prereq_masteries=prereq_masteries,
                            task_type=_p.get("task_type", "procedural"),
                            n_steps=_p.get("n_steps", 1),
                        )
                        selected_task = chosen
                        selected_kc = kc_entry
                        selected_context = ctx
                        selected_source = "diagnostic_cat"
                        selected_irt_fallback = irt_fallback_used
                        break
                elif diag_kc != kc_entry.kc_id:
                    continue

            # 6. Фаза 1 (первые PHASE1_TASK_THRESHOLD заданий): grade-based эвристика.
            # Система ещё не знает ученика — используем ZPD без бандита.
            if phase1:
                from .selector import select_task as _select_task_heuristic
                target_difficulty = compute_zpd_target_difficulty(kc_entry.mastery_effective, difficulty_mode)
                stretch = min(0.95, kc_entry.mastery_effective + 0.4)
                chosen, _ = _select_task_heuristic(tasks, target_difficulty, mastery=kc_entry.mastery_effective, stretch_difficulty=stretch, rng=_rng)
                _p = (chosen.get("parts") or [{}])[0]
                ctx = _build_context(
                    mastery_kc=kc_entry.mastery_effective,
                    grade=grade,
                    avg_reward=0.0,
                    count=0,
                    errors_streak=consecutive_errors.get(kc_entry.kc_id, 0),
                    prereq_masteries=prereq_masteries,
                    task_type=_p.get("task_type", "procedural"),
                    n_steps=_p.get("n_steps", 1),
                )
                selected_task = chosen
                selected_kc = kc_entry
                selected_context = ctx
                selected_source = "phase1_heuristic"
                selected_irt_fallback = irt_fallback_used
                break

            # 7. Control group: эвристика (ближайшая сложность)
            if variant == "control":
                from .selector import select_task as _select_task_heuristic
                target_difficulty = compute_zpd_target_difficulty(kc_entry.mastery_effective, difficulty_mode)
                stretch = min(0.95, kc_entry.mastery_effective + 0.4)
                chosen, _ = _select_task_heuristic(tasks, target_difficulty, mastery=kc_entry.mastery_effective, stretch_difficulty=stretch, rng=_rng)
                _p = (chosen.get("parts") or [{}])[0]
                ctx = _build_context(
                    mastery_kc=kc_entry.mastery_effective,
                    grade=grade,
                    avg_reward=0.0,
                    count=0,
                    errors_streak=consecutive_errors.get(kc_entry.kc_id, 0),
                    prereq_masteries=prereq_masteries,
                    task_type=_p.get("task_type", "procedural"),
                    n_steps=_p.get("n_steps", 1),
                )
                selected_task = chosen
                selected_kc = kc_entry
                selected_context = ctx
                selected_source = "control_heuristic"
                selected_irt_fallback = irt_fallback_used
                break

            # 7. Treatment: cluster stats + Thompson Sampling с exploration
            task_ids = [str(t["task_id"]) for t in tasks]
            if cluster_id is not None:
                stats = await _get_cluster_task_stats(cluster_id, task_ids)
            else:
                stats = {}

            roll = _rng.random()
            mode_cluster_rate = mode_p["cluster_explore"]
            mode_epsilon_rate = mode_p["epsilon"]

            # ── Cluster exploration ─────────────────────────────────────────
            underexplored = [
                t for t in tasks
                if stats.get(str(t["task_id"]), (0.0, 0))[1] < CLUSTER_EXPLORE_THRESHOLD
            ]
            if underexplored and roll < mode_cluster_rate:
                chosen = _rng.choice(underexplored)
                avg_r, cnt = stats.get(str(chosen["task_id"]), (0.0, 0))
                _p = (chosen.get("parts") or [{}])[0]
                ctx = _build_context(
                    mastery_kc=kc_entry.mastery_effective,
                    grade=grade,
                    avg_reward=avg_r,
                    count=cnt,
                    errors_streak=consecutive_errors.get(kc_entry.kc_id, 0),
                    prereq_masteries=prereq_masteries,
                    task_type=_p.get("task_type", "procedural"),
                    n_steps=_p.get("n_steps", 1),
                )
                selected_task = chosen
                selected_kc = kc_entry
                selected_context = ctx
                selected_source = "cluster_exploration"
                selected_irt_fallback = irt_fallback_used
                break

            # ── ε-greedy fallback ────────────────────────────────────────────
            elif roll < mode_cluster_rate + mode_epsilon_rate:
                chosen = _rng.choice(tasks)
                avg_r, cnt = stats.get(str(chosen["task_id"]), (0.0, 0))
                _p = (chosen.get("parts") or [{}])[0]
                ctx = _build_context(
                    mastery_kc=kc_entry.mastery_effective,
                    grade=grade,
                    avg_reward=avg_r,
                    count=cnt,
                    errors_streak=consecutive_errors.get(kc_entry.kc_id, 0),
                    prereq_masteries=prereq_masteries,
                    task_type=_p.get("task_type", "procedural"),
                    n_steps=_p.get("n_steps", 1),
                )
                selected_task = chosen
                selected_kc = kc_entry
                selected_context = ctx
                selected_source = "exploration"
                selected_irt_fallback = irt_fallback_used
                break

            # ── Thompson Sampling exploitation ───────────────────────────────
            else:
                effective_cluster = cluster_id if cluster_id is not None else GLOBAL_CLUSTER_ID
                async with AsyncSessionLocal() as db:
                    model = await _load_student_model(db, req.student_id, kc_entry.kc_id, effective_cluster)
                    best_task = None
                    best_score = -float("inf")
                    best_context = None

                    for task in tasks:
                        tid = str(task["task_id"])
                        avg_reward, count = stats.get(tid, (0.0, 0))
                        _tp = (task.get("parts") or [{}])[0]
                        x = _build_context(
                            mastery_kc=kc_entry.mastery_effective,
                            grade=grade,
                            avg_reward=avg_reward,
                            count=count,
                            errors_streak=consecutive_errors.get(kc_entry.kc_id, 0),
                            prereq_masteries=prereq_masteries,
                            task_type=_tp.get("task_type", "procedural"),
                            n_steps=_tp.get("n_steps", 1),
                        )
                        s = model.score(x)
                        # ZPD accuracy bonus: штраф за задания далёкие от целевых 65% успеха
                        irt_diff = task.get("parts", [{}])[0].get("irt_difficulty") or 0.5
                        p_correct = compute_p_correct(kc_entry.mastery_effective, irt_diff)
                        s -= 0.5 * abs(p_correct - TARGET_ZPD_ACCURACY)
                        # Bridge bonus: задание-мостик к следующему шагу плана
                        secondary_kcs = task.get("parts", [{}])[0].get("secondary_kcs", [])
                        if next_plan_kc_id and next_plan_kc_id in secondary_kcs:
                            s += 0.1
                        if s > best_score:
                            best_score = s
                            best_task = task
                            best_context = x

                selected_task = best_task
                selected_kc = kc_entry
                selected_context = best_context
                selected_source = "thompson_sampling"
                selected_irt_fallback = irt_fallback_used
                break

        if selected_task is None or selected_kc is None:
            logger.info("recommendation failed: no tasks available for student=%s", req.student_id)
            raise HTTPException(
                status_code=404,
                detail="Нет доступных заданий — все пройдены или TaskBank пуст",
            )

    # 7a. Save CAT state if in diagnostic phase
    if in_cat and cat_state is not None:
        cat_done = cat_state.is_complete
        await _save_cat_state(req.student_id, cat_state, completed=cat_done)
        if cat_done:
            logger.info("Diagnostic CAT completed for student=%s, %d tasks used", req.student_id, cat_state.tasks_used)

    # 7b. Teacher alert при fallback с plan KC
    if plan_fallback_occurred:
        async with AsyncSessionLocal() as db:
            await db.execute(
                sa.text("""
                    INSERT INTO teacher_alerts
                        (id, student_id, plan_id, kc_id, alert_type, mastery_at_alert,
                         tasks_spent, message, created_at)
                    VALUES
                        (:id, :student_id,
                         (SELECT id FROM learning_plans WHERE student_id = :student_id
                          ORDER BY created_at DESC LIMIT 1),
                         :kc_id, 'content_gap_plan', :mastery,
                         0, :message, :now)
                """),
                {
                    "id": uuid.uuid4(),
                    "student_id": req.student_id,
                    "kc_id": active_plan_kc_id,
                    "mastery": mastery.get(active_plan_kc_id, 0.0),
                    "message": f"Plan KC '{active_plan_kc_id}' has no available tasks — fallback to ZPD",
                    "now": datetime.now(timezone.utc),
                },
            )
            await db.commit()

    # 8. Записываем в bandit_log + инкрементируем tasks_spent активного шага плана
    if plan_fallback_occurred:
        selection_reason = "plan_lock_fallback"
    context_list = selected_context.tolist() if selected_context is not None else [0.0] * CONTEXT_DIM
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text(
                "INSERT INTO bandit_log"
                " (id, student_id, task_id, kc_id, context_vector, reward, recommended_at,"
                "  selection_reason, exploration_type, zpd_candidates_count,"
                "  plan_step_id, difficulty_mode, fallback_occurred, irt_fallback_occurred)"
                " VALUES (:id, :student_id, :task_id, :kc_id, :context_vector, NULL, :recommended_at,"
                "  :selection_reason, :exploration_type, :zpd_candidates_count,"
                "  :plan_step_id, :difficulty_mode, :fallback_occurred, :irt_fallback_occurred)"
            ),
            {
                "id": uuid.uuid4(),
                "student_id": req.student_id,
                "task_id": selected_task["task_id"],
                "kc_id": selected_kc.kc_id,
                "context_vector": context_list,
                "recommended_at": datetime.now(timezone.utc),
                "selection_reason": selection_reason,
                "exploration_type": selected_source,
                "zpd_candidates_count": len(candidates),
                "plan_step_id": active_plan_step_id,
                "difficulty_mode": difficulty_mode,
                "fallback_occurred": plan_fallback_occurred,
                "irt_fallback_occurred": selected_irt_fallback,
            },
        )
        # Инкрементируем tasks_spent для активного шага плана
        spent_row = (await db.execute(
            sa.text("""
                UPDATE plan_steps
                   SET tasks_spent = tasks_spent + 1
                 WHERE status = 'in_progress'
                   AND kc_id = :kc_id
                   AND plan_id = (
                       SELECT id FROM learning_plans
                       WHERE student_id = :student_id
                       ORDER BY created_at DESC LIMIT 1
                   )
                RETURNING tasks_spent
            """),
            {"student_id": req.student_id, "kc_id": selected_kc.kc_id},
        )).fetchone()
        await db.commit()

    tasks_spent_now = int(spent_row[0]) if spent_row else 0
    mastery_current = mastery.get(selected_kc.kc_id, 0.0)

    # 9. MicroSummary — плановый триггер (каждые 15 задач) + OnFrustration
    # Для студентов без плана используем общий task_count как proxy
    effective_spent = tasks_spent_now if tasks_spent_now > 0 else task_count + 1
    if effective_spent > 0:
        summary = await compute_micro_summary(
            req.student_id,
            selected_kc.kc_id,
            mastery_current,
            effective_spent,
        )
        if (effective_spent % 15 == 0
                or is_frustrated(summary["frustration_count"], summary["velocity"])):
            await publish_micro_summary(summary)

    half_life_map = {"arithmetic": 90.0, "algebra": 45.0, "geometry": 60.0, "statistics": 30.0}
    half_life = half_life_map.get(selected_kc.subject, 45.0)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "recommendation student=%s kc=%s source=%s phase1=%s variant=%s latency_ms=%.1f",
        req.student_id,
        selected_kc.kc_id,
        selected_source,
        phase1,
        variant,
        elapsed_ms,
    )

    return RecommendResponse(
        task_id=str(selected_task["task_id"]),
        kc_id=selected_kc.kc_id,
        subject=selected_kc.subject,
        recommendation_source=selected_source,
        half_life_days=half_life,
        task=selected_task,
    )


@app.patch("/bandit_log/reward", status_code=200)
async def update_bandit_reward(req: UpdateRewardRequest):
    """
    Заполняет reward в bandit_log и обновляет Thompson Sampling модель.
    Вызывается из gateway после получения ответа ученика.
    SELECT FOR UPDATE на bandit_model предотвращает гонку при конкурентных обновлениях.
    """
    answered_at = req.answered_at or datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        # Получаем запись bandit_log и обновляем reward
        row = (await db.execute(
            sa.text("""
                UPDATE bandit_log
                   SET reward = :reward,
                       raw_score = COALESCE(:raw_score, raw_score),
                       hints_used = COALESCE(:hints_used, hints_used),
                       time_spent_seconds = COALESCE(:time_spent_seconds, time_spent_seconds),
                       irt_difficulty = COALESCE(:irt_difficulty, irt_difficulty),
                       mastery_delta = COALESCE(:mastery_delta, mastery_delta),
                       answered_at = :answered_at
                 WHERE id = (
                     SELECT id FROM bandit_log
                     WHERE student_id = :student_id AND task_id = :task_id
                     ORDER BY recommended_at DESC LIMIT 1
                 )
                RETURNING kc_id, context_vector
            """),
            {
                "reward": req.reward,
                "raw_score": req.raw_score,
                "hints_used": req.hints_used,
                "time_spent_seconds": req.time_spent_seconds,
                "irt_difficulty": req.irt_difficulty,
                "mastery_delta": req.mastery_delta,
                "answered_at": answered_at,
                "student_id": req.student_id,
                "task_id": req.task_id,
            },
        )).fetchone()

        if not row:
            await db.rollback()
            logger.warning(
                "bandit reward update skipped: missing log entry for student=%s task=%s",
                req.student_id,
                req.task_id,
            )
            raise HTTPException(status_code=404, detail="bandit_log entry not found")

        kc_id = row[0]
        context_vector = row[1]   # list[float] из PostgreSQL ARRAY

        # Инкрементально обновляем cluster_task_stats (running average)
        # чтобы x[7]/x[8] в context vector не были нулями между запусками кластеризации
        await db.execute(
            sa.text("""
                INSERT INTO cluster_task_stats
                    (cluster_id, task_id, avg_reward, interaction_count, computed_at)
                SELECT
                    sc.cluster_id,
                    :task_id,
                    :reward,
                    1,
                    :now
                FROM student_clusters sc
                WHERE sc.student_id = :student_id
                ON CONFLICT (cluster_id, task_id) DO UPDATE
                    SET avg_reward = (
                            cluster_task_stats.avg_reward * cluster_task_stats.interaction_count
                            + EXCLUDED.avg_reward
                        ) / (cluster_task_stats.interaction_count + 1),
                        interaction_count = cluster_task_stats.interaction_count + 1,
                        computed_at = EXCLUDED.computed_at
            """),
            {
                "task_id": req.task_id,
                "reward": req.reward,
                "student_id": req.student_id,
                "now": answered_at,
            },
        )

        # Update CAT state if in diagnostic phase
        cat_state, cat_completed = await _get_cat_state(req.student_id)
        if cat_state is not None and not cat_completed:
            if req.raw_score is not None and req.irt_difficulty is not None:
                update_cat_state(cat_state, kc_id, score=req.raw_score, irt_difficulty=req.irt_difficulty)
            else:
                n = cat_state.kc_n.get(kc_id, 0)
                cat_state.kc_n[kc_id] = n + 1
                cat_state.tasks_used += 1
            await _save_cat_state(req.student_id, cat_state, completed=cat_state.is_complete)
            if cat_state.is_complete:
                async with httpx.AsyncClient(trust_env=False) as http:
                    try:
                        await clients.apply_diagnostic_mastery(
                            http,
                            req.student_id,
                            cat_state.to_mastery(),
                        )
                    except httpx.HTTPStatusError as e:
                        logger.warning("failed to apply CAT mastery for student=%s: %s", req.student_id, e)

        # Обновляем Thompson Sampling только для treatment-группы
        variant = await _get_experiment_variant(req.student_id)
        if variant == "treatment":
            student_cluster = await _get_cluster_id(req.student_id)
            effective_cluster = student_cluster if student_cluster is not None else GLOBAL_CLUSTER_ID
            x = np.array(context_vector, dtype=np.float64)
            if len(x) != CONTEXT_DIM:
                x = np.pad(x, (0, max(0, CONTEXT_DIM - len(x))))[:CONTEXT_DIM]

            # Обновляем личную модель студента
            student_model = await _load_student_model(db, req.student_id, kc_id, effective_cluster)
            student_model.update(x, req.reward)
            await _save_student_model(db, req.student_id, student_model)

            # Обновляем кластерную модель — кластер продолжает учиться на всех студентах
            cluster_model = await _load_model(db, kc_id, effective_cluster)
            cluster_model.update(x, req.reward)
            await _save_model(db, cluster_model)

        await db.commit()

    logger.info(
        "bandit reward updated student=%s task=%s reward=%.4f",
        req.student_id,
        req.task_id,
        req.reward,
    )
    return {"updated": True}
