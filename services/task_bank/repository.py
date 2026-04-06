"""Операции с БД для сервиса банка заданий."""

import uuid
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import TaskModel, PartModel


class TaskRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_task(self, task_id: uuid.UUID) -> TaskModel | None:
        result = await self.db.execute(
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .options(selectinload(TaskModel.parts))
        )
        return result.scalar_one_or_none()

    async def get_tasks_by_kc(
        self,
        kc_id: str,
        grade_min: int,
        exclude_task_ids: list[uuid.UUID] | None = None,
        limit: int = 20,
    ) -> list[TaskModel]:
        """
        Задания где kc_id есть в primary_kcs любой из частей.
        Исключает уже виденные учеником задания.
        """
        exclude = exclude_task_ids or []
        rows = await self.db.execute(
            text("""
                SELECT DISTINCT t.task_id
                FROM tasks t
                JOIN parts p ON p.task_id = t.task_id
                WHERE :kc_id = ANY(p.primary_kcs)
                  AND t.grade_min <= :grade_min
                  AND t.task_id != ALL(:exclude)
                ORDER BY t.task_id
                LIMIT :limit
            """),
            {
                "kc_id": kc_id,
                "grade_min": grade_min,
                "exclude": [str(tid) for tid in exclude],
                "limit": limit,
            },
        )
        task_ids = [row[0] for row in rows]
        if not task_ids:
            return []

        result = await self.db.execute(
            select(TaskModel)
            .where(TaskModel.task_id.in_(task_ids))
            .options(selectinload(TaskModel.parts))
        )
        return list(result.scalars().all())

    async def create_task(self, task: TaskModel) -> TaskModel:
        self.db.add(task)
        await self.db.flush()
        return task

    async def count_tasks_by_kc(self, kc_id: str) -> int:
        rows = await self.db.execute(
            text("""
                SELECT COUNT(DISTINCT t.task_id)
                FROM tasks t
                JOIN parts p ON p.task_id = t.task_id
                WHERE :kc_id = ANY(p.primary_kcs)
            """),
            {"kc_id": kc_id},
        )
        return rows.scalar_one()
