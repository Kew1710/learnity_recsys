"""Сервис подбора задания (Retrieval).

Горячий путь:
  1. Получить mastery + seen_tasks из Profile
  2. Получить ZPD-кандидатов из Graph
  3. Выбрать KC (subject rotation)
  4. Получить задания для KC из TaskBank (с exclude фильтром)
  5. Выбрать задание — LinUCB с гибридным контекстом
  6. Записать в bandit_log (context_vector, reward=NULL)
  7. Вернуть задание + метаданные рекомендации

После ответа (PATCH /bandit_log/reward):
  8. Обновить reward в bandit_log
  9. Обновить LinUCB-модель (A, b) в bandit_model
"""

import math
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.db import AsyncSessionLocal
from . import clients
from .linucb import LinUCBModel, CONTEXT_DIM, GLOBAL_CLUSTER_ID
from .selector import ZPDEntry, select_kc_from_zpd, compute_zpd_target_difficulty, compute_p_correct, TARGET_ZPD_ACCURACY
from .micro_summary import compute_micro_summary, should_publish_summary, is_frustrated
from .kafka_producer import publish_micro_summary
from services.clustering.cluster import assign_cluster_for_new_student, save_student_cluster

# ---------------------------------------------------------------------------
# Exploration / Cooldown constants
# ---------------------------------------------------------------------------

CLUSTER_EXPLORE_RATE = 0.20       # 20%: выбрать задание с interaction_count < порога
CLUSTER_EXPLORE_THRESHOLD = 3     # задание «неисследовано» если видели < 3 раз в кластере
EPSILON_GREEDY_RATE = 0.05        # 5% fallback: полностью случайное задание
KC_COOLDOWN_WINDOW = 6            # окно последних рекомендаций для проверки cooldown
KC_COOLDOWN_MAX = 3               # если KC ≥ N раз за окно → пропустить (cooldown)

PHASE1_TASK_THRESHOLD = 15        # до этого порога — grade-based эвристика (фаза исследования)

_rng = random.Random()

app = FastAPI(title="Retrieval Service")


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
    recommendation_source: str           # "linucb"
    half_life_days: float                # передаётся в profile при submit
    task: dict                           # полные данные задания


class UpdateRewardRequest(BaseModel):
    student_id: uuid.UUID
    task_id: uuid.UUID
    reward: float
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
) -> tuple[dict[str, float], str, str | None, str | None]:
    """
    Возвращает:
      - {kc_id: priority} для активных шагов плана
      - difficulty_mode активного шага (in_progress) или 'build' если нет
      - next_plan_kc_id — kc_id следующего pending шага (для bridge bonus) или None
      - active_plan_kc_id — kc_id текущего in_progress шага или None (для hard lock)
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT ps.kc_id, ps.priority, ps.status, ps.difficulty_mode
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

    in_progress_seen = False
    for r in rows:
        kc_id, priority, status, dm = r[0], float(r[1]), r[2], r[3]
        priorities[kc_id] = priority
        if status == "in_progress" and not in_progress_seen:
            difficulty_mode = dm or "build"
            active_plan_kc_id = kc_id
            in_progress_seen = True
        elif status == "pending" and in_progress_seen and next_plan_kc_id is None:
            next_plan_kc_id = kc_id

    return priorities, difficulty_mode, next_plan_kc_id, active_plan_kc_id


async def _get_cluster_id(student_id: uuid.UUID) -> int | None:
    """Возвращает cluster_id ученика или None если ещё не назначен."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("SELECT cluster_id FROM student_clusters WHERE student_id = :sid"),
            {"sid": student_id},
        )).fetchone()
    return int(row[0]) if row else None


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
) -> LinUCBModel:
    """
    Загружает личную LinUCB-модель студента.
    При первом обращении инициализируется из кл��стерной модели (transfer learning).
    """
    row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM student_bandit_model"
            " WHERE student_id = :sid AND kc_id = :kc_id FOR UPDATE"
        ),
        {"sid": student_id, "kc_id": kc_id},
    )).fetchone()
    if row:
        return LinUCBModel.from_bytes(kc_id, fallback_cluster_id, bytes(row[0]), bytes(row[1]))

    # Первое обращение — копируем кластерную модель как prior
    cluster_row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM bandit_model"
            " WHERE cluster_id = :cluster_id AND kc_id = :kc_id"
        ),
        {"cluster_id": fallback_cluster_id, "kc_id": kc_id},
    )).fetchone()
    if cluster_row:
        return LinUCBModel.from_bytes(kc_id, fallback_cluster_id, bytes(cluster_row[0]), bytes(cluster_row[1]))
    return LinUCBModel.init(kc_id, fallback_cluster_id)


async def _save_student_model(db, student_id: uuid.UUID, model: LinUCBModel) -> None:
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


async def _load_model(db, kc_id: str, cluster_id: int) -> LinUCBModel:
    """Загружает LinUCB-модель кластера из БД. Создаёт новую если не существует."""
    row = (await db.execute(
        sa.text(
            "SELECT a_matrix, b_vector FROM bandit_model"
            " WHERE cluster_id = :cluster_id AND kc_id = :kc_id FOR UPDATE"
        ),
        {"cluster_id": cluster_id, "kc_id": kc_id},
    )).fetchone()
    if row:
        return LinUCBModel.from_bytes(kc_id, cluster_id, bytes(row[0]), bytes(row[1]))
    return LinUCBModel.init(kc_id, cluster_id)


async def _save_model(db, model: LinUCBModel) -> None:
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
    Подбирает следующее задание для ученика через LinUCB.
    Проходит по ZPD-кандидатам до первого KC у которого есть задания в TaskBank.
    """
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
        consecutive_errors: dict[str, int] = {
            kc_id: v.get("consecutive_errors", 0)
            for kc_id, v in mastery_raw.items()
        }

        # 2. Seen tasks (только за последние 90 дней — задания старше TTL повторяются)
        try:
            seen_tasks = await clients.get_seen_tasks(http, req.student_id, ttl_days=90)
        except httpx.HTTPStatusError:
            seen_tasks = []

        # 3. ZPD кандидаты
        try:
            zpd_raw = await clients.get_zpd(http, mastery, grade)
        except httpx.HTTPStatusError:
            raise HTTPException(status_code=502, detail="Graph service error")

        if not zpd_raw:
            raise HTTPException(status_code=404, detail="ZPD пуст — нет подходящих тем")

        # Приоритеты из учебного плана + difficulty_mode + next_plan_kc + текущий шаг
        plan_priorities, difficulty_mode, next_plan_kc_id, active_plan_kc_id = await _get_plan_info(req.student_id)

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
        # KC Cooldown не применяется к шагу плана (нужна концентрация, не ротация).
        # Fallback на ZPD если KC шага плана нет среди кандидатов (например, задания закончились).
        if active_plan_kc_id:
            plan_step_candidates = [c for c in candidates if c.kc_id == active_plan_kc_id]
            ordered = plan_step_candidates if plan_step_candidates else candidates
        else:
            preferred = select_kc_from_zpd(candidates, req.last_subject)
            ordered = [preferred] + [c for c in candidates if c.kc_id != preferred.kc_id] if preferred else candidates

        # 5. A/B вариант — определяем алгоритм выбора задания
        variant = await _get_experiment_variant(req.student_id)

        # 5b. Фаза студента и кластер
        task_count = await _get_student_task_count(req.student_id)
        phase1 = task_count < PHASE1_TASK_THRESHOLD   # True → grade-based эвристика

        cluster_id = await _get_cluster_id(req.student_id) if variant == "treatment" else None

        # Граница фаз 1→2: переназначаем кластер по реальному поведению
        if variant == "treatment" and task_count == PHASE1_TASK_THRESHOLD:
            new_cluster = assign_cluster_for_new_student(mastery)
            if new_cluster is not None:
                await save_student_cluster(req.student_id, new_cluster)
                cluster_id = new_cluster

        # 5c. Cooldown: последние KC этого ученика
        recent_kcs = await _get_recent_kcs(req.student_id)

        selected_task = None
        selected_kc: ZPDEntry | None = None
        selected_context: np.ndarray | None = None
        selected_source: str = "linucb"

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
                continue

            # Пре-реквизиты KC для контекстного вектора LinUCB (x[1..4])
            try:
                prereqs = await clients.get_kc_prerequisites(http, kc_entry.kc_id)
                prereq_masteries = [mastery.get(p["kc_id"], 0.0) for p in prereqs]
            except httpx.HTTPStatusError:
                prereq_masteries = []

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
                break

            # 7. Treatment: cluster stats + LinUCB с exploration
            task_ids = [str(t["task_id"]) for t in tasks]
            if cluster_id is not None:
                stats = await _get_cluster_task_stats(cluster_id, task_ids)
            else:
                stats = {}

            roll = _rng.random()

            # ── Cluster exploration (20%) ────────────────────────────────────
            # Показываем задание, которое нашим кластером ещё почти не изучено.
            # Это умный exploration: заполняем белые пятна кооперативно.
            underexplored = [
                t for t in tasks
                if stats.get(str(t["task_id"]), (0.0, 0))[1] < CLUSTER_EXPLORE_THRESHOLD
            ]
            if underexplored and roll < CLUSTER_EXPLORE_RATE:
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
                break

            # ── ε-greedy fallback (5%) ───────────────────────────────────────
            # Полностью случайное задание из пула — если кластер пустой или
            # как базовый стохастический слой.
            elif roll < CLUSTER_EXPLORE_RATE + EPSILON_GREEDY_RATE:
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
                break

            # ── LinUCB exploitation (75%) ────────────────────────────────────
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
                selected_source = "linucb"
                break

        if selected_task is None or selected_kc is None:
            raise HTTPException(
                status_code=404,
                detail="Нет доступных заданий — все пройдены или TaskBank пуст",
            )

    # 8. Записываем в bandit_log + инкрементируем tasks_spent активного шага плана
    context_list = selected_context.tolist() if selected_context is not None else [0.0] * CONTEXT_DIM
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text(
                "INSERT INTO bandit_log (id, student_id, task_id, kc_id, context_vector, reward, recommended_at)"
                " VALUES (:id, :student_id, :task_id, :kc_id, :context_vector, NULL, :recommended_at)"
            ),
            {
                "id": uuid.uuid4(),
                "student_id": req.student_id,
                "task_id": selected_task["task_id"],
                "kc_id": selected_kc.kc_id,
                "context_vector": context_list,
                "recommended_at": datetime.now(timezone.utc),
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
    if tasks_spent_now > 0:
        summary = await compute_micro_summary(
            req.student_id,
            selected_kc.kc_id,
            mastery_current,
            tasks_spent_now,
        )
        if (tasks_spent_now % 15 == 0
                or is_frustrated(summary["frustration_count"], summary["velocity"])):
            await publish_micro_summary(summary)

    half_life_map = {"arithmetic": 90.0, "algebra": 45.0, "geometry": 60.0, "statistics": 30.0}
    half_life = half_life_map.get(selected_kc.subject, 45.0)

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
    Заполняет reward в bandit_log и обновляет LinUCB-модель.
    Вызывается из gateway после получения ответа ученика.
    SELECT FOR UPDATE на bandit_model предотвращает гонку при конкурентных обновлениях.
    """
    answered_at = req.answered_at or datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        # Получаем запись bandit_log и обновляем reward
        row = (await db.execute(
            sa.text("""
                UPDATE bandit_log
                   SET reward = :reward, answered_at = :answered_at
                 WHERE id = (
                     SELECT id FROM bandit_log
                     WHERE student_id = :student_id AND task_id = :task_id
                     ORDER BY recommended_at DESC LIMIT 1
                 )
                RETURNING kc_id, context_vector
            """),
            {
                "reward": req.reward,
                "answered_at": answered_at,
                "student_id": req.student_id,
                "task_id": req.task_id,
            },
        )).fetchone()

        if not row:
            await db.rollback()
            raise HTTPException(status_code=404, detail="bandit_log entry not found")

        kc_id = row[0]
        context_vector = row[1]   # list[float] из PostgreSQL ARRAY

        # Инкрементально обновляем cluster_task_stats (running average)
        # чтобы x[7]/x[8] в LinUCB не были нулями между запусками кластеризации
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

        # Обновляем LinUCB только для treatment-группы
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

    return {"updated": True}
