"""Сервис банка заданий."""

import uuid
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db import get_db
from .models import TaskModel, PartModel
from .repository import TaskRepository

app = FastAPI(title="Task Bank Service")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PartResponse(BaseModel):
    part_id: str
    description: str
    primary_kcs: list[str]
    secondary_kcs: list[str]
    answer_type: str
    correct_answer: Optional[object] = None
    tolerance: Optional[float] = None
    irt_difficulty: Optional[float] = None
    irt_discrimination: Optional[float] = None
    irt_guessing: Optional[float] = None
    task_type: str = "procedural"
    n_steps: int = 1
    scaffolding_steps: list[str]


class TaskResponse(BaseModel):
    task_id: uuid.UUID
    grade_min: int
    source: str
    parts: list[PartResponse]


def _task_to_response(task: TaskModel) -> TaskResponse:
    return TaskResponse(
        task_id=task.task_id,
        grade_min=task.grade_min,
        source=task.source,
        parts=[
            PartResponse(
                part_id=p.part_id,
                description=p.description,
                primary_kcs=p.primary_kcs,
                secondary_kcs=p.secondary_kcs or [],
                answer_type=p.answer_type,
                correct_answer=p.correct_answer,
                tolerance=p.tolerance,
                irt_difficulty=p.irt_difficulty,
                irt_discrimination=p.irt_discrimination,
                irt_guessing=p.irt_guessing,
                task_type=p.task_type,
                n_steps=p.n_steps,
                scaffolding_steps=p.scaffolding_steps or [],
            )
            for p in task.parts
        ],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ВАЖНО: /tasks/by_kc и /tasks/by_kc/count должны быть выше /tasks/{task_id},
# иначе FastAPI интерпретирует "by_kc" как task_id.

@app.get("/tasks/by_kc/count")
async def count_tasks_by_kc(kc_id: str = Query(...), db: AsyncSession = Depends(get_db)):
    repo = TaskRepository(db)
    count = await repo.count_tasks_by_kc(kc_id)
    return {"kc_id": kc_id, "count": count}


@app.get("/tasks/by_kc", response_model=list[TaskResponse])
async def get_tasks_by_kc(
    kc_id: str = Query(...),
    grade_min: int = Query(...),
    exclude: list[uuid.UUID] = Query(default=[]),
    limit: int = Query(default=10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Задания для указанной KC.
    exclude — task_id уже виденных учеником заданий (фильтр пузыря).
    """
    repo = TaskRepository(db)
    tasks = await repo.get_tasks_by_kc(
        kc_id=kc_id,
        grade_min=grade_min,
        exclude_task_ids=list(exclude),
        limit=limit,
    )
    return [_task_to_response(t) for t in tasks]


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    repo = TaskRepository(db)
    task = await repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)
