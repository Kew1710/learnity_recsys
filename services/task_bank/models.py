"""SQLAlchemy ORM модели банка заданий."""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db import Base


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    grade_min: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False, default="bank")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    parts: Mapped[list["PartModel"]] = relationship("PartModel", back_populates="task", cascade="all, delete-orphan")


class PartModel(Base):
    __tablename__ = "parts"

    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.task_id", ondelete="CASCADE"), primary_key=True)
    part_id: Mapped[str] = mapped_column(String, primary_key=True)
    description: Mapped[str] = mapped_column(String, nullable=False)

    primary_kcs: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    secondary_kcs: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)

    answer_type: Mapped[str] = mapped_column(String, nullable=False)
    correct_answer = mapped_column(JSONB, nullable=True)
    tolerance: Mapped[float | None] = mapped_column(Float, nullable=True)

    irt_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    irt_discrimination: Mapped[float | None] = mapped_column(Float, nullable=True)
    irt_guessing: Mapped[float | None] = mapped_column(Float, nullable=True)

    # procedural | conceptual | word_problem | mixed
    task_type: Mapped[str] = mapped_column(String, nullable=False, server_default="procedural")
    n_steps: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    scaffolding_steps: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    distractors_map = mapped_column(JSONB, nullable=False, default=dict)

    task: Mapped["TaskModel"] = relationship("TaskModel", back_populates="parts")
