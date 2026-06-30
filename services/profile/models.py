"""SQLAlchemy ORM модели для сервиса профиля."""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db import Base


class StudentModel(Base):
    __tablename__ = "students"

    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    grade: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    guessing_rate: Mapped[float] = mapped_column(Float, default=0.0)
    hint_dependence: Mapped[float] = mapped_column(Float, default=0.0)
    review_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    estimated_lr: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)

    mastery: Mapped[list["MasteryModel"]] = relationship("MasteryModel", back_populates="student", cascade="all, delete-orphan")
    interactions: Mapped[list["InteractionModel"]] = relationship("InteractionModel", back_populates="student", cascade="all, delete-orphan")


class MasteryModel(Base):
    __tablename__ = "mastery"

    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), primary_key=True)
    kc_id: Mapped[str] = mapped_column(String, primary_key=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    last_practiced: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    attempts_count: Mapped[int] = mapped_column(Integer, default=0)
    p_transit: Mapped[float] = mapped_column(Float, default=0.1)
    p_slip: Mapped[float] = mapped_column(Float, default=0.1)
    p_guess: Mapped[float] = mapped_column(Float, default=0.2)
    consecutive_errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_correct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    recent_accuracy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    student: Mapped["StudentModel"] = relationship("StudentModel", back_populates="mastery")


class LearningPlanModel(Base):
    __tablename__ = "learning_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    goal_type: Mapped[str | None] = mapped_column(String, nullable=True)
    mastery_threshold: Mapped[float] = mapped_column(Float, default=0.80, nullable=False)
    require_test: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    coverage_variant: Mapped[str | None] = mapped_column(String, nullable=True)
    task_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)

    steps: Mapped[list["PlanStepModel"]] = relationship("PlanStepModel", back_populates="plan", cascade="all, delete-orphan", order_by="PlanStepModel.priority.desc()")


class PlanStepModel(Base):
    __tablename__ = "plan_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("learning_plans.id", ondelete="CASCADE"), nullable=False)
    kc_id: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    reason: Mapped[str] = mapped_column(String, nullable=False)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    difficulty_mode: Mapped[str] = mapped_column(String, nullable=False, default="build")
    tasks_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    tasks_spent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    plan: Mapped["LearningPlanModel"] = relationship("LearningPlanModel", back_populates="steps")


class InteractionModel(Base):
    __tablename__ = "interactions"

    interaction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    part_id: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    time_spent_seconds: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    misconception_triggered: Mapped[str | None] = mapped_column(String, nullable=True)
    recommendation_source: Mapped[str | None] = mapped_column(String, nullable=True)

    student: Mapped["StudentModel"] = relationship("StudentModel", back_populates="interactions")
