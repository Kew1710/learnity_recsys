from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnswerType(str, Enum):
    numeric = "numeric"
    multiple_choice = "multiple_choice"
    multi_select = "multi_select"
    proof = "proof"


class TaskSource(str, Enum):
    bank = "bank"
    generated = "generated"


# ---------------------------------------------------------------------------
# IRT параметры задания
# ---------------------------------------------------------------------------

class IRT(BaseModel):
    difficulty: float       # b — порог сложности
    discrimination: float   # a — различающая способность
    guessing: float         # c — вероятность угадать


# ---------------------------------------------------------------------------
# Часть задания
# ---------------------------------------------------------------------------

class Part(BaseModel):
    part_id: str
    description: str

    primary_kcs: list[str]              # минимум одна KC
    secondary_kcs: list[str] = []

    answer_type: AnswerType
    correct_answer: Optional[Any] = None    # None для proof
    tolerance: Optional[float] = None       # только для numeric

    irt: Optional[IRT] = None               # None до IRT-калибровки

    scaffolding_steps: list[str] = []
    distractors_map: dict[str, str] = {}    # неправильный ответ → misconception_id

    @field_validator("primary_kcs")
    @classmethod
    def primary_kcs_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("primary_kcs не может быть пустым — укажи хотя бы одну KC")
        return v


# ---------------------------------------------------------------------------
# Задание
# ---------------------------------------------------------------------------

class Task(BaseModel):
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    grade_min: int
    source: TaskSource = TaskSource.bank
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parts: list[Part]

    @model_validator(mode="after")
    def validate_multipart_for_multi_kc(self) -> "Task":
        """
        Задание с >2 уникальными primary KC должно быть многочастным.

        Правило: если ученик ошибается в однозначном ответе, система не может
        понять в какой из тем была ошибка. Многочастный формат решает это —
        каждая часть тестирует конкретную KC и даёт точный BKT-сигнал.

        Допустимо:
          - 1 часть, 1-2 primary KC
          - N частей, любое число primary KC (каждая часть — отдельный сигнал)

        Запрещено:
          - 1 часть, >2 уникальных primary KC суммарно
        """
        if len(self.parts) == 1:
            all_primary = set(self.parts[0].primary_kcs)
            if len(all_primary) > 2:
                raise ValueError(
                    f"Задание затрагивает {len(all_primary)} primary KC "
                    f"({', '.join(sorted(all_primary))}) но состоит из одной части. "
                    "Разбей на несколько частей чтобы система могла точно определить "
                    "где ученик ошибся."
                )
        return self


# ---------------------------------------------------------------------------
# Mastery по одному KC
# ---------------------------------------------------------------------------

class MasteryRecord(BaseModel):
    probability: float          # P(mastered) — хранимое значение
    last_practiced: datetime
    attempts_count: int = 0
    p_transit: float = 0.1      # P(T) — вероятность выучить за попытку
    p_slip: float = 0.1         # P(S) — ошибиться зная KC
    p_guess: float = 0.2        # P(G) — угадать не зная KC


# ---------------------------------------------------------------------------
# Поведенческий профиль ученика
# ---------------------------------------------------------------------------

class BehavioralProfile(BaseModel):
    guessing_rate: float = 0.0
    hint_dependence: float = 0.0


# ---------------------------------------------------------------------------
# Профиль ученика
# ---------------------------------------------------------------------------

class Student(BaseModel):
    student_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    grade: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    behavioral: BehavioralProfile = Field(default_factory=BehavioralProfile)
    mastery: dict[str, MasteryRecord] = {}  # kc_id → MasteryRecord


# ---------------------------------------------------------------------------
# Лог взаимодействий
# ---------------------------------------------------------------------------

class Interaction(BaseModel):
    interaction_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    student_id: uuid.UUID
    task_id: uuid.UUID
    part_id: str
    score: float                            # 0.0–1.0
    hints_used: int = 0
    time_spent_seconds: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    misconception_triggered: Optional[str] = None


# ---------------------------------------------------------------------------
# Kafka события
# ---------------------------------------------------------------------------

class AnswerSubmittedEvent(BaseModel):
    event_type: str = "answer_submitted"
    student_id: uuid.UUID
    task_id: uuid.UUID
    interactions: list[Interaction]     # одно на каждый part


class MasteryUpdatedEvent(BaseModel):
    event_type: str = "mastery_updated"
    student_id: uuid.UUID
    updated_kcs: list[str]              # kc_id которые обновились
