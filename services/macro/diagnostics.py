"""
Diagnostic reason layer v1.

Determines the likely cause of a student's struggle on a KC,
replacing pure symptom-based reactions with cause-aware decisions.

Possible diagnoses:
  prereq_gap        — weak prerequisite is holding the student back
  content_gap       — too few tasks at the appropriate difficulty
  uncertain_estimate — not enough data to trust the mastery estimate
  regression        — student previously knew this but is forgetting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DiagnosisType = Literal["prereq_gap", "content_gap", "uncertain_estimate", "regression", "on_track"]


@dataclass
class Diagnosis:
    reason: DiagnosisType
    confidence: float  # 0..1 how sure we are about this diagnosis
    detail: str


def diagnose(
    *,
    mastery_current: float,
    velocity: float,
    frustration_count: int,
    avg_score: float,
    tasks_spent: int,
    attempts_count: int,
    mastery_confidence: float,
    weakest_prereq_mastery: float | None,
    task_count_for_kc: int | None,
) -> Diagnosis:
    """
    Determine the most likely reason for the student's current state on a KC.

    Args:
        mastery_current: current mastery estimate
        velocity: learning velocity from MicroSummary
        frustration_count: consecutive low rewards
        avg_score: average reward over window
        tasks_spent: tasks spent on this KC step
        attempts_count: total attempts on this KC (from mastery record)
        mastery_confidence: confidence in mastery estimate (from П1.1)
        weakest_prereq_mastery: mastery of the weakest strong prerequisite, or None
        task_count_for_kc: number of available tasks for this KC, or None if unknown
    """
    # Not struggling — on track
    if frustration_count < 2 and velocity >= 0 and avg_score >= 0.45:
        return Diagnosis("on_track", 0.9, "student is progressing normally")

    # Uncertain estimate: too few observations to trust mastery
    if mastery_confidence < 0.3 and attempts_count < 5:
        return Diagnosis(
            "uncertain_estimate", 0.8,
            f"only {attempts_count} attempts, confidence={mastery_confidence:.2f} — "
            "mastery estimate unreliable",
        )

    # Content gap: very few tasks available
    if task_count_for_kc is not None and task_count_for_kc < 3:
        return Diagnosis(
            "content_gap", 0.9,
            f"only {task_count_for_kc} tasks available for this KC",
        )

    # Prereq gap: weak prerequisite
    if weakest_prereq_mastery is not None and weakest_prereq_mastery < 0.5:
        return Diagnosis(
            "prereq_gap", 0.85,
            f"weakest prereq mastery={weakest_prereq_mastery:.2f} — "
            "foundation insufficient",
        )

    # Regression: mastery was high but performance dropped
    if mastery_current >= 0.7 and avg_score < 0.4 and frustration_count >= 2:
        return Diagnosis(
            "regression", 0.75,
            f"mastery={mastery_current:.2f} but avg_score={avg_score:.2f} — "
            "likely forgetting or overestimated mastery",
        )

    # Prereq gap (weaker signal): prereq exists but mastery is mediocre
    if weakest_prereq_mastery is not None and weakest_prereq_mastery < 0.65:
        return Diagnosis(
            "prereq_gap", 0.6,
            f"weakest prereq mastery={weakest_prereq_mastery:.2f} — "
            "may be contributing to difficulty",
        )

    # Default: content gap if we can't determine the cause
    if frustration_count >= 2:
        return Diagnosis(
            "content_gap", 0.4,
            "frustration detected but cause unclear — possible task quality issue",
        )

    return Diagnosis("on_track", 0.5, "no strong signal of a specific problem")
