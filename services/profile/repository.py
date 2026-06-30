"""Операции с БД для сервиса профиля."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import StudentModel, MasteryModel, InteractionModel


class StudentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, student_id: uuid.UUID, grade: int, review_mode: bool = False) -> StudentModel:
        student = StudentModel(student_id=student_id, grade=grade, review_mode=review_mode)
        self.db.add(student)
        await self.db.flush()
        return student

    async def update_review_mode(self, student_id: uuid.UUID, review_mode: bool) -> StudentModel | None:
        student = await self.get(student_id)
        if student:
            student.review_mode = review_mode
            await self.db.flush()
        return student

    async def get(self, student_id: uuid.UUID) -> StudentModel | None:
        result = await self.db.execute(
            select(StudentModel)
            .where(StudentModel.student_id == student_id)
            .options(selectinload(StudentModel.mastery))
        )
        return result.scalar_one_or_none()

    async def get_mastery(self, student_id: uuid.UUID, kc_id: str) -> MasteryModel | None:
        result = await self.db.execute(
            select(MasteryModel).where(
                MasteryModel.student_id == student_id,
                MasteryModel.kc_id == kc_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_all_mastery(self, student_id: uuid.UUID) -> list[MasteryModel]:
        result = await self.db.execute(
            select(MasteryModel).where(MasteryModel.student_id == student_id)
        )
        return list(result.scalars().all())

    async def upsert_mastery(
        self,
        student_id: uuid.UUID,
        kc_id: str,
        probability: float,
        now: datetime,
        p_transit: float,
        p_slip: float,
        p_guess: float,
        score: float = 1.0,
        recent_accuracy: float = 0.0,
    ) -> MasteryModel:
        record = await self.get_mastery(student_id, kc_id)
        if record is None:
            record = MasteryModel(
                student_id=student_id,
                kc_id=kc_id,
                probability=probability,
                last_practiced=now,
                attempts_count=1,
                p_transit=p_transit,
                p_slip=p_slip,
                p_guess=p_guess,
                consecutive_errors=0 if score >= 0.5 else 1,
                consecutive_correct=1 if score >= 0.5 else 0,
                recent_accuracy=recent_accuracy,
            )
            self.db.add(record)
        else:
            record.probability = probability
            record.last_practiced = now
            record.attempts_count += 1
            record.consecutive_errors = 0 if score >= 0.5 else record.consecutive_errors + 1
            record.consecutive_correct = record.consecutive_correct + 1 if score >= 0.5 else 0
            record.recent_accuracy = recent_accuracy
        await self.db.flush()
        return record

    async def get_seen_tasks(
        self,
        student_id: uuid.UUID,
        ttl_days: int | None = None,
    ) -> list[uuid.UUID]:
        q = select(InteractionModel.task_id).where(
            InteractionModel.student_id == student_id
        )
        if ttl_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            q = q.where(InteractionModel.timestamp >= cutoff)
        result = await self.db.execute(q.distinct())
        return list(result.scalars().all())

    async def bulk_initialize_mastery(
        self,
        student_id: uuid.UUID,
        kc_probabilities: dict[str, float],  # kc_id → initial probability
        now: datetime,
    ) -> int:
        """
        Batch INSERT ... ON CONFLICT DO NOTHING для cold start.
        Не перезаписывает KC где mastery уже выставлена.
        Возвращает количество вставленных записей.
        """
        if not kc_probabilities:
            return 0
        rows = [
            {
                "student_id": str(student_id),
                "kc_id": kc_id,
                "probability": prob,
                "last_practiced": now,
                "attempts_count": 0,
                "p_transit": 0.1,
                "p_slip": 0.1,
                "p_guess": 0.2,
            }
            for kc_id, prob in kc_probabilities.items()
        ]
        await self.db.execute(
            text("""
                INSERT INTO mastery
                    (student_id, kc_id, probability, last_practiced, attempts_count,
                     p_transit, p_slip, p_guess)
                VALUES
                    (:student_id, :kc_id, :probability, :last_practiced, :attempts_count,
                     :p_transit, :p_slip, :p_guess)
                ON CONFLICT (student_id, kc_id) DO NOTHING
            """),
            rows,
        )
        await self.db.flush()
        return len(rows)

    async def save_interaction(self, interaction: InteractionModel) -> None:
        self.db.add(interaction)
        await self.db.flush()
