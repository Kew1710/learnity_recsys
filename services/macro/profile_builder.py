"""Build and refresh MacroStudentProfile from current student data."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import sqlalchemy as sa

from shared.db import AsyncSessionLocal
from . import clients
from .prereq_extractor import extract_prereq_subgraph
from .student_profile import (
    MacroStudentProfile,
    get_macro_student_profile,
    save_macro_student_profile,
)

RECENT_WINDOW = 20
GLOBAL_WINDOW = 60
LOW_CONFIDENCE_THRESHOLD = 0.35


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def derive_pacing_mode(
    uncertainty_level: float,
    frustration_risk: float,
    learning_speed_recent: float,
    recovery_after_error: float,
) -> str:
    if uncertainty_level >= 0.55 or frustration_risk >= 0.45:
        return "careful"
    if learning_speed_recent >= 0.06 and recovery_after_error >= 0.04:
        return "aggressive"
    return "balanced"


def derive_budget_multiplier(
    pacing_mode: str,
    uncertainty_level: float,
    weak_prereq_fraction: float,
) -> float:
    base = {"careful": 1.25, "balanced": 1.0, "aggressive": 0.85}[pacing_mode]
    return round(base + 0.35 * uncertainty_level + 0.25 * weak_prereq_fraction, 3)


def derive_prereq_strictness(weak_prereq_fraction: float, frustration_risk: float) -> float:
    return round(clamp(0.35 + 0.45 * weak_prereq_fraction + 0.25 * frustration_risk), 3)


def derive_test_readiness_bias(
    mastery_confidence_mean: float,
    recovery_after_error: float,
    frustration_risk: float,
) -> float:
    return round(clamp(0.5 + 0.35 * mastery_confidence_mean + 0.2 * recovery_after_error - 0.3 * frustration_risk), 3)


def derive_step_granularity(
    pacing_mode: str,
    uncertainty_level: float,
    weak_prereq_fraction: float,
) -> float:
    base = {"careful": 0.8, "balanced": 1.0, "aggressive": 1.15}[pacing_mode]
    return round(max(0.5, base - 0.2 * uncertainty_level - 0.15 * weak_prereq_fraction), 3)


def build_profile_from_inputs(
    student_id: uuid.UUID,
    *,
    target_kc_id: str | None,
    detailed_mastery: dict[str, dict],
    target_subgraph_nodes: list[str],
    learning_speed_global: float,
    learning_speed_recent: float,
    frustration_risk: float,
    recovery_after_error: float,
) -> MacroStudentProfile:
    relevant_nodes = target_subgraph_nodes or list(detailed_mastery.keys())
    relevant_details = [detailed_mastery[kc] for kc in relevant_nodes if kc in detailed_mastery]
    all_details = list(detailed_mastery.values())

    confidence_values = [float(v.get("confidence", 0.0)) for v in relevant_details]
    mastery_values = [float(v.get("probability_effective", 0.0)) for v in relevant_details]
    attempts_values = [int(v.get("attempts_count", 0)) for v in relevant_details]

    uncertainty_level = round(
        safe_mean([1.0 if v.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD else 0.0 for v in all_details], default=1.0),
        3,
    )
    mastery_confidence_mean = round(safe_mean(confidence_values, default=0.0), 3)
    weak_prereq_fraction = round(
        safe_mean([1.0 if float(v.get("probability_effective", 0.0)) < 0.6 else 0.0 for v in relevant_details], default=0.0),
        3,
    )
    target_subgraph_mastery_mean = round(safe_mean(mastery_values, default=0.0), 3)
    mean_attempts = safe_mean([float(v) for v in attempts_values], default=0.0)

    tasks_to_gain_01_mastery = round(
        max(1.0, 0.1 / max(learning_speed_recent, 0.01)),
        3,
    )
    recovery_after_error = round(clamp(recovery_after_error, 0.0, 1.0), 3)
    frustration_risk = round(clamp(frustration_risk, 0.0, 1.0), 3)
    stall_risk_baseline = round(clamp(0.35 * uncertainty_level + 0.35 * frustration_risk + 0.3 * weak_prereq_fraction), 3)
    regression_risk_baseline = round(clamp(0.35 * (1.0 - mastery_confidence_mean) + 0.25 * uncertainty_level + 0.1 * max(0.0, 1.0 - learning_speed_global)), 3)

    pacing_mode = derive_pacing_mode(
        uncertainty_level=uncertainty_level,
        frustration_risk=frustration_risk,
        learning_speed_recent=learning_speed_recent,
        recovery_after_error=recovery_after_error,
    )

    return MacroStudentProfile(
        student_id=student_id,
        version=1,
        updated_at=datetime.now(timezone.utc),
        target_kc_id=target_kc_id,
        uncertainty_level=uncertainty_level,
        mastery_confidence_mean=mastery_confidence_mean,
        weak_prereq_fraction=weak_prereq_fraction,
        target_subgraph_mastery_mean=target_subgraph_mastery_mean,
        learning_speed_global=round(max(0.0, learning_speed_global), 4),
        learning_speed_recent=round(max(0.0, learning_speed_recent), 4),
        tasks_to_gain_01_mastery=tasks_to_gain_01_mastery,
        recovery_after_error=recovery_after_error,
        frustration_risk=frustration_risk,
        stall_risk_baseline=stall_risk_baseline,
        regression_risk_baseline=regression_risk_baseline,
        pacing_mode=pacing_mode,
        budget_multiplier=derive_budget_multiplier(pacing_mode, uncertainty_level, weak_prereq_fraction),
        prereq_strictness=derive_prereq_strictness(weak_prereq_fraction, frustration_risk),
        test_readiness_bias=derive_test_readiness_bias(
            mastery_confidence_mean,
            recovery_after_error,
            frustration_risk,
        ),
        step_granularity=derive_step_granularity(pacing_mode, uncertainty_level, weak_prereq_fraction),
    )


async def build_initial_macro_profile(
    student_id: uuid.UUID,
    target_kc_id: str | None = None,
) -> MacroStudentProfile:
    async with httpx.AsyncClient(trust_env=False) as http:
        detailed_mastery = await clients.get_student_mastery_detailed(http, student_id)
        subgraph_nodes = await _load_target_subgraph_nodes(http, target_kc_id, detailed_mastery)

    global_rows = await _load_bandit_rows(student_id, GLOBAL_WINDOW)
    recent_rows = global_rows[:RECENT_WINDOW]

    profile = build_profile_from_inputs(
        student_id,
        target_kc_id=target_kc_id,
        detailed_mastery=detailed_mastery,
        target_subgraph_nodes=subgraph_nodes,
        learning_speed_global=_mean_mastery_delta(global_rows),
        learning_speed_recent=_mean_mastery_delta(recent_rows),
        frustration_risk=_compute_frustration_risk(recent_rows),
        recovery_after_error=_compute_recovery_after_error(recent_rows),
    )
    return await save_macro_student_profile(profile)


async def refresh_macro_profile(
    student_id: uuid.UUID,
    target_kc_id: str | None = None,
) -> MacroStudentProfile:
    existing = await get_macro_student_profile(student_id)
    effective_target = target_kc_id or (existing.target_kc_id if existing else None)
    return await build_initial_macro_profile(student_id, effective_target)


async def _load_target_subgraph_nodes(
    http: httpx.AsyncClient,
    target_kc_id: str | None,
    detailed_mastery: dict[str, dict],
) -> list[str]:
    if not target_kc_id:
        return list(detailed_mastery.keys())

    graph: dict[str, list[dict]] = {}
    visited: set[str] = set()
    queue = [target_kc_id]
    while queue:
        kc_id = queue.pop()
        if kc_id in visited:
            continue
        visited.add(kc_id)
        try:
            prereqs = await clients.get_all_prerequisites(http, kc_id)
        except httpx.HTTPError:
            prereqs = []
        graph[kc_id] = prereqs
        for prereq in prereqs:
            if prereq["kc_id"] not in visited:
                queue.append(prereq["kc_id"])

    mastery = {
        kc_id: float(detail.get("probability_effective", 0.0))
        for kc_id, detail in detailed_mastery.items()
    }
    subgraph = extract_prereq_subgraph(target_kc_id, mastery, graph)
    return subgraph["nodes"] or [target_kc_id]


async def _load_bandit_rows(student_id: uuid.UUID, window: int) -> list[dict]:
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                sa.text(
                    """
                    SELECT raw_score, mastery_delta, hints_used
                    FROM bandit_log
                    WHERE student_id = :student_id
                    ORDER BY recommended_at DESC
                    LIMIT :window
                    """
                ),
                {"student_id": student_id, "window": window},
            )
        ).fetchall()

    return [
        {
            "raw_score": float(row[0]) if row[0] is not None else None,
            "mastery_delta": float(row[1]) if row[1] is not None else None,
            "hints_used": int(row[2] or 0),
        }
        for row in rows
    ]


def _mean_mastery_delta(rows: list[dict]) -> float:
    deltas = [row["mastery_delta"] for row in rows if row.get("mastery_delta") is not None]
    if not deltas:
        return 0.03
    return round(safe_mean(deltas, default=0.03), 4)


def _compute_frustration_risk(rows: list[dict]) -> float:
    if not rows:
        return 0.5
    frustrated = 0
    for row in rows:
        score = row.get("raw_score")
        mastery_delta = row.get("mastery_delta")
        hints_used = row.get("hints_used", 0)
        if score is None:
            continue
        if score < 0.45 and hints_used > 0 and (mastery_delta is None or mastery_delta <= 0.01):
            frustrated += 1
    return round(frustrated / max(1, len(rows)), 4)


def _compute_recovery_after_error(rows: list[dict]) -> float:
    if len(rows) < 2:
        return 0.03
    recovered: list[float] = []
    chronological = list(reversed(rows))
    for idx, row in enumerate(chronological[:-1]):
        score = row.get("raw_score")
        if score is None or score >= 0.5:
            continue
        next_row = chronological[idx + 1]
        delta = next_row.get("mastery_delta")
        if delta is not None:
            recovered.append(delta)
    if not recovered:
        return 0.03
    return round(max(0.0, safe_mean(recovered, default=0.03)), 4)
