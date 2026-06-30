"""Unit tests for diagnostic CAT module."""

import math
from services.retrieval.diagnostic_cat import (
    CATState, select_diagnostic_kc, select_diagnostic_task, update_cat_state,
    CAT_BUDGET,
)


def test_cat_state_from_mastery():
    mastery = {"kc_add": 0.3, "kc_mul": 0.7, "kc_div": 0.5}
    state = CATState.from_mastery(mastery)
    assert set(state.kc_theta.keys()) == {"kc_add", "kc_mul", "kc_div"}
    assert state.tasks_used == 0
    assert not state.is_complete


def test_cat_state_to_mastery_roundtrip():
    mastery = {"kc_add": 0.3, "kc_mul": 0.7}
    state = CATState.from_mastery(mastery)
    recovered = state.to_mastery()
    for kc in mastery:
        assert abs(recovered[kc] - mastery[kc]) < 0.01


def test_cat_state_completes_at_budget():
    state = CATState(kc_theta={"a": 0.0}, kc_n={"a": 0}, tasks_used=CAT_BUDGET)
    assert state.is_complete


def test_cat_state_completes_early_if_enough_kcs_tested():
    state = CATState(
        kc_theta={"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
        kc_n={"a": 2, "b": 2, "c": 2, "d": 0},
        tasks_used=6,
    )
    assert state.is_complete


def test_select_diagnostic_kc_prefers_untested():
    state = CATState(
        kc_theta={"kc_a": 0.0, "kc_b": 0.0, "kc_c": 0.0},
        kc_n={"kc_a": 3, "kc_b": 0, "kc_c": 1},
    )
    kc = select_diagnostic_kc(state, ["kc_a", "kc_b", "kc_c"])
    assert kc == "kc_b"


def test_select_diagnostic_kc_breaks_ties_by_uncertainty():
    state = CATState(
        kc_theta={"kc_a": 2.0, "kc_b": 0.1},
        kc_n={"kc_a": 0, "kc_b": 0},
    )
    kc = select_diagnostic_kc(state, ["kc_a", "kc_b"])
    assert kc == "kc_b"


def test_select_diagnostic_task_targets_p50():
    tasks = [
        {"task_id": "1", "parts": [{"irt_difficulty": -2.0}]},
        {"task_id": "2", "parts": [{"irt_difficulty": 0.0}]},
        {"task_id": "3", "parts": [{"irt_difficulty": 2.0}]},
    ]
    chosen = select_diagnostic_task(tasks, theta=0.0)
    assert chosen["task_id"] == "2"


def test_select_diagnostic_task_adapts_to_theta():
    tasks = [
        {"task_id": "easy", "parts": [{"irt_difficulty": -1.0}]},
        {"task_id": "medium", "parts": [{"irt_difficulty": 1.0}]},
        {"task_id": "hard", "parts": [{"irt_difficulty": 3.0}]},
    ]
    chosen = select_diagnostic_task(tasks, theta=1.0)
    assert chosen["task_id"] == "medium"


def test_update_cat_state():
    state = CATState(
        kc_theta={"kc_a": 0.0},
        kc_n={"kc_a": 0},
        tasks_used=0,
    )
    update_cat_state(state, "kc_a", score=1.0, irt_difficulty=0.0)
    assert state.kc_n["kc_a"] == 1
    assert state.tasks_used == 1
    assert state.kc_theta["kc_a"] > 0.0


def test_update_cat_state_correct_decreases_after_wrong():
    state = CATState(
        kc_theta={"kc_a": 0.0},
        kc_n={"kc_a": 0},
        tasks_used=0,
    )
    update_cat_state(state, "kc_a", score=0.0, irt_difficulty=0.0)
    assert state.kc_theta["kc_a"] < 0.0
