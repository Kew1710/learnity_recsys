"""Macro Planner Service — управление учебными планами."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.db import AsyncSessionLocal
from . import clients
from .prereq_extractor import extract_prereq_subgraph
from .tasks_estimator import estimate as estimate_tasks
from .policy_mode1 import train_policy, policy_path, save_policy, load_policy, SubgraphQAgent
from .policy_mode2 import rank_coverage_kcs
from .plan_lifecycle import (
    PlanStep,
    create_plan,
    evaluate_micro_summary,
    apply_plan_actions,
    advance_step,
    check_test_phase,
)
from .kafka_consumer import run_consumer


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_consumer())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Macro Planner Service", lifespan=lifespan)

MACRO_URL = "http://127.0.0.1:8006"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    student_id: uuid.UUID
    mode: str                            # "target_mastery" | "coverage"
    target_kc_id: Optional[str] = None  # Режим 1
    mastery_threshold: float = 0.80
    require_test: bool = False
    coverage_variant: Optional[str] = None  # Режим 2: count|mass|frontier
    task_budget: Optional[int] = None
    cluster_id: Optional[int] = None
    n_train_episodes: int = 500          # уменьшено для быстрого запуска


class PlanResponse(BaseModel):
    plan_id: str
    steps_count: int


class EvaluateRequest(BaseModel):
    summary: dict


class EvaluateResponse(BaseModel):
    actions: list[dict]
    applied: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/plans", response_model=PlanResponse, status_code=201)
async def create_new_plan(req: PlanRequest):
    """
    Создаёт учебный план для ученика.

    Режим 1 (target_mastery):
      1. Загружает mastery ученика
      2. BFS-строит подграф пре-реквизитов
      3. Обучает Q-агента (или загружает кэшированного)
      4. Разворачивает план: последовательность KC по политике
    """
    if req.mode == "target_mastery" and not req.target_kc_id:
        raise HTTPException(status_code=422, detail="target_kc_id обязателен для режима target_mastery")

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            mastery = await clients.get_student_mastery(http, req.student_id)
            student_grade = await clients.get_student_grade(http, req.student_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Student not found")
            raise HTTPException(status_code=502, detail="Profile service error")

        # Проверяем доступность target KC для текущего класса студента.
        # ZPD допускает KC от (grade-2) до (grade+1). Выше — недостижимо.
        if req.mode == "target_mastery" and req.target_kc_id:
            target_kc_grade = await clients.get_kc_grade(http, req.target_kc_id)
            if target_kc_grade is not None and target_kc_grade > student_grade + 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"KC '{req.target_kc_id}' (grade {target_kc_grade}) недостижима для студента "
                        f"grade {student_grade}: ZPD допускает до grade {student_grade + 1}"
                    ),
                )

        if req.mode == "target_mastery":
            steps = await _build_mode1_plan(
                http, req, mastery, student_grade
            )
        elif req.mode == "coverage":
            steps = await _build_mode2_plan(http, req, mastery)
        else:
            raise HTTPException(status_code=422, detail=f"Неизвестный режим: {req.mode}")

    if not steps:
        raise HTTPException(status_code=422, detail="Не удалось построить план — подграф пуст или KC уже освоена")

    plan_id = await create_plan(
        student_id=req.student_id,
        mode=req.mode,
        params={
            "title": f"{req.mode}: {req.target_kc_id or 'coverage'}",
            "mastery_threshold": req.mastery_threshold,
            "require_test": req.require_test,
            "coverage_variant": req.coverage_variant,
            "task_budget": req.task_budget,
        },
        steps=steps,
    )

    return PlanResponse(plan_id=str(plan_id), steps_count=len(steps))


@app.get("/plans/{plan_id}")
async def get_plan(plan_id: uuid.UUID):
    """Возвращает статус плана и текущий активный шаг."""
    async with AsyncSessionLocal() as db:
        plan_row = (await db.execute(
            sa.text("SELECT id, student_id, title, goal_type, created_at FROM learning_plans WHERE id = :id"),
            {"id": plan_id},
        )).fetchone()

        if not plan_row:
            raise HTTPException(status_code=404, detail="Plan not found")

        steps = (await db.execute(
            sa.text("""
                SELECT kc_id, priority, status, difficulty_mode, tasks_budget, tasks_spent, reason
                FROM plan_steps WHERE plan_id = :plan_id ORDER BY priority DESC
            """),
            {"plan_id": plan_id},
        )).fetchall()

    return {
        "plan_id": str(plan_row[0]),
        "student_id": str(plan_row[1]),
        "title": plan_row[2],
        "goal_type": plan_row[3],
        "created_at": plan_row[4].isoformat(),
        "steps": [
            {
                "kc_id": s[0],
                "priority": s[1],
                "status": s[2],
                "difficulty_mode": s[3],
                "tasks_budget": s[4],
                "tasks_spent": s[5],
                "reason": s[6],
            }
            for s in steps
        ],
    }


@app.post("/plans/{plan_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_plan(plan_id: uuid.UUID, req: EvaluateRequest):
    """
    Ручной триггер пересмотра плана на основе MicroSummary.
    Используется в тестах и как fallback если Kafka недоступна.
    """
    async with AsyncSessionLocal() as db:
        plan_row = (await db.execute(
            sa.text("SELECT student_id FROM learning_plans WHERE id = :id"),
            {"id": plan_id},
        )).fetchone()

    if not plan_row:
        raise HTTPException(status_code=404, detail="Plan not found")

    student_id = plan_row[0]
    actions = evaluate_micro_summary(req.summary)

    await apply_plan_actions(student_id, actions)

    return EvaluateResponse(
        actions=[{"type": a.action_type, "payload": a.payload} for a in actions],
        applied=True,
    )


@app.post("/plans/{plan_id}/advance")
async def advance_plan(plan_id: uuid.UUID):
    """Завершает текущий шаг и активирует следующий."""
    async with AsyncSessionLocal() as db:
        plan_row = (await db.execute(
            sa.text("SELECT student_id FROM learning_plans WHERE id = :id"),
            {"id": plan_id},
        )).fetchone()

    if not plan_row:
        raise HTTPException(status_code=404, detail="Plan not found")

    has_next = await advance_step(plan_row[0])
    return {"has_next_step": has_next}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _build_mode1_plan(
    http: httpx.AsyncClient,
    req: PlanRequest,
    mastery: dict[str, float],
    student_grade: int,
) -> list[PlanStep]:
    """Строит шаги плана для Режима 1 через SubgraphQAgent."""
    # 1. Получить граф пре-реквизитов
    visited: set[str] = set()
    queue = [req.target_kc_id]
    graph: dict[str, list[dict]] = {}

    while queue:
        kc_id = queue.pop()
        if kc_id in visited:
            continue
        visited.add(kc_id)
        try:
            prereqs = await clients.get_all_prerequisites(http, kc_id)
        except httpx.HTTPStatusError:
            prereqs = []
        graph[kc_id] = prereqs
        for p in prereqs:
            if p["kc_id"] not in visited:
                queue.append(p["kc_id"])

    subgraph = extract_prereq_subgraph(
        target_kc_id=req.target_kc_id,
        mastery=mastery,
        graph=graph,
        threshold=req.mastery_threshold,
    )

    if not subgraph["nodes"]:
        return []

    # 2. Загрузить или обучить политику
    p_path = policy_path(req.cluster_id or 0, req.target_kc_id)
    try:
        agent = load_policy(p_path)
    except (FileNotFoundError, Exception):
        agent = train_policy(
            subgraph=subgraph,
            target_kc_id=req.target_kc_id,
            initial_mastery=mastery,
            target_mastery=req.mastery_threshold,
            n_episodes=req.n_train_episodes,
        )
        try:
            save_policy(agent, p_path)
        except Exception:
            pass  # директория models/ может отсутствовать в тестах

    # 3. Загрузить grade-карту и развернуть план
    try:
        kc_grade_map = await clients.get_all_kc_grades(http)
    except Exception:
        kc_grade_map = {}

    return _rollout_plan(agent, subgraph, mastery, req, student_grade, kc_grade_map)


def _rollout_plan(
    agent: SubgraphQAgent,
    subgraph: dict,
    mastery: dict[str, float],
    req: PlanRequest,
    student_grade: int,
    kc_grade_map: dict[str, int],
) -> list[PlanStep]:
    """
    Генерирует упорядоченный список KC по жадной политике (epsilon=0).
    Порядок: сначала все пре-реквизиты (по политике), затем target KC последним.
    Фильтрует KC по ZPD: grade-2 <= kc_grade <= grade+1.
    """
    def is_grade_eligible(kc_id: str) -> bool:
        kc_grade = kc_grade_map.get(kc_id, student_grade)
        return student_grade - 2 <= kc_grade <= student_grade + 1

    seen: set[str] = set()
    steps: list[PlanStep] = []
    current_mastery = dict(mastery)

    needs_work = {
        n for n in subgraph["nodes"]
        if current_mastery.get(n, 0.0) < req.mastery_threshold and is_grade_eligible(n)
    }
    prereqs_pool = [n for n in needs_work if n != req.target_kc_id]
    target_needs_work = req.target_kc_id in needs_work

    priority = 1.0

    # 1. Развернуть пре-реквизиты через политику
    max_steps = len(prereqs_pool) + 1
    for _ in range(max_steps):
        remaining = [a for a in prereqs_pool if a not in seen]
        if not remaining:
            break
        kc_id = agent.select_action(current_mastery, remaining, epsilon=0.0)
        seen.add(kc_id)

        tasks_budget = estimate_tasks(
            m_current=current_mastery.get(kc_id, 0.0),
            m_target=req.mastery_threshold,
            cluster_avg=None,
            n_practiced_in_subject=0,
            n_sims=100,
        )
        steps.append(PlanStep(
            kc_id=kc_id,
            difficulty_mode="build",
            tasks_budget=tasks_budget,
            reason=f"prereq rollout, mastery={current_mastery.get(kc_id, 0.0):.2f}",
            priority=round(priority, 2),
        ))
        priority -= 0.05

    # 2. Target KC всегда последним шагом
    if target_needs_work:
        kc_id = req.target_kc_id
        m = current_mastery.get(kc_id, 0.0)
        tasks_budget = estimate_tasks(
            m_current=m,
            m_target=req.mastery_threshold,
            cluster_avg=None,
            n_practiced_in_subject=0,
            n_sims=100,
        )
        difficulty_mode = "test" if check_test_phase(m, req.mastery_threshold) else "build"
        steps.append(PlanStep(
            kc_id=kc_id,
            difficulty_mode=difficulty_mode,
            tasks_budget=tasks_budget,
            reason=f"target kc, mastery={m:.2f}",
            priority=round(priority, 2),
        ))

    return steps


async def _build_mode2_plan(
    http: httpx.AsyncClient,
    req: PlanRequest,
    mastery: dict[str, float],
) -> list[PlanStep]:
    """
    Строит план для Режима 2 (Coverage).

    Скоринговая функция ранжирует KC по приоритету:
      - grade_introduced <= student_grade  (только доступные темы)
      - prereq mastery                     (готовность пре-реквизитов)
      - gap до порога / вариант coverage   (count / mass / frontier)

    Возвращает топ-N шагов с равным бюджетом.
    """
    import asyncio

    budget = req.task_budget or 100
    variant = req.coverage_variant or "count"

    try:
        grade = await clients.get_student_grade(http, req.student_id)
    except httpx.HTTPStatusError:
        grade = 8

    # Все KC графа с их классом
    all_kcs_resp = await http.get(f"{clients.GRAPH_URL}/kcs", timeout=clients.TIMEOUT)
    all_kcs_resp.raise_for_status()
    all_kcs = [kc for kc in all_kcs_resp.json() if kc["grade_introduced"] <= grade]

    if not all_kcs:
        return []

    # Пре-реквизиты для всех KC — параллельные запросы
    async def fetch_prereq_masteries(kc_id: str) -> list[float]:
        try:
            prereqs = await clients.get_all_prerequisites(http, kc_id)
            return [mastery.get(p["kc_id"], 0.0) for p in prereqs]
        except Exception:
            return []

    prereq_lists = await asyncio.gather(
        *[fetch_prereq_masteries(kc["kc_id"]) for kc in all_kcs]
    )

    kcs_with_prereqs = [
        {**kc, "prereq_masteries": pm}
        for kc, pm in zip(all_kcs, prereq_lists)
    ]

    top_kcs = rank_coverage_kcs(
        kcs_with_prereqs,
        mastery,
        student_grade=grade,
        variant=variant,
        budget_ratio=1.0,
        top_n=10,
    )

    if not top_kcs:
        return []

    kc_budget = max(1, budget // len(top_kcs))
    return [
        PlanStep(
            kc_id=kc_id,
            difficulty_mode="build",
            tasks_budget=kc_budget,
            reason=f"coverage/{variant}, budget={budget}",
            priority=round(1.0 - i * 0.05, 2),
        )
        for i, kc_id in enumerate(top_kcs)
    ]
