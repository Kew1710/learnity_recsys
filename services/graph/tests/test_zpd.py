"""
Юнит-тесты ZPD-логики. Без Neo4j — только чистые функции.
"""

import pytest
from services.graph.zpd import (
    compute_zpd_candidates,
    is_grade_eligible,
    KCNode,
    Prerequisite,
    MASTERY_THRESHOLD,
    MASTERY_CEILING,
)


def make_kc(kc_id: str, grade: int, difficulty: float = 0.5) -> KCNode:
    return KCNode(kc_id=kc_id, grade_introduced=grade, difficulty_base=difficulty)


def make_prereq(kc_id: str, strength: float = 0.9) -> Prerequisite:
    return Prerequisite(kc_id=kc_id, strength=strength)


# ---------------------------------------------------------------------------
# is_grade_eligible
# ---------------------------------------------------------------------------

class TestGradeEligible:
    def test_current_grade_ok(self):
        assert is_grade_eligible(8, 8) is True

    def test_one_below_ok(self):
        assert is_grade_eligible(7, 8) is True

    def test_two_below_ok(self):
        assert is_grade_eligible(6, 8) is True

    def test_three_below_filtered(self):
        assert is_grade_eligible(5, 8) is False

    def test_one_above_ok(self):
        assert is_grade_eligible(9, 8) is True

    def test_two_above_filtered(self):
        assert is_grade_eligible(10, 8) is False


# ---------------------------------------------------------------------------
# compute_zpd_candidates
# ---------------------------------------------------------------------------

class TestZPD:
    def _run(self, kcs, prereqs, mastery, grade=8):
        return compute_zpd_candidates(kcs, prereqs, mastery, grade)

    # --- grade filter ---

    def test_too_old_kc_excluded(self):
        kcs = [make_kc("kc_old", grade=4)]
        result = self._run(kcs, {}, {}, grade=8)
        assert result == []

    def test_too_advanced_kc_excluded(self):
        kcs = [make_kc("kc_future", grade=10)]
        result = self._run(kcs, {}, {}, grade=8)
        assert result == []

    def test_one_grade_above_included(self):
        kcs = [make_kc("kc_ahead", grade=9)]
        result = self._run(kcs, {}, {}, grade=8)
        assert len(result) == 1

    # --- mastery ceiling ---

    def test_mastered_kc_excluded(self):
        kcs = [make_kc("kc_done", grade=8)]
        result = self._run(kcs, {}, {"kc_done": MASTERY_CEILING}, grade=8)
        assert result == []

    def test_almost_mastered_still_included(self):
        kcs = [make_kc("kc_almost", grade=8)]
        result = self._run(kcs, {}, {"kc_almost": MASTERY_CEILING - 0.01}, grade=8)
        assert len(result) == 1

    def test_no_mastery_record_included(self):
        kcs = [make_kc("kc_new", grade=8)]
        result = self._run(kcs, {}, {}, grade=8)
        assert len(result) == 1

    # --- prerequisites ---

    def test_unmet_prereq_included_but_not_ready(self):
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_prereq", grade=7),
        ]
        prereqs = {"kc_target": [make_prereq("kc_prereq", strength=0.9)]}
        mastery = {"kc_prereq": 0.3}
        result = self._run(kcs, prereqs, mastery, grade=8)
        assert len(result) == 2
        target = next(c for c in result if c.kc_id == "kc_target")
        assert target.unmet_prereqs == ["kc_prereq"]

    def test_met_prereq_is_ready(self):
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_prereq", grade=7),
        ]
        prereqs = {"kc_target": [make_prereq("kc_prereq", strength=0.9)]}
        mastery = {"kc_prereq": MASTERY_THRESHOLD + 0.01}
        result = self._run(kcs, prereqs, mastery, grade=8)
        target = next(c for c in result if c.kc_id == "kc_target")
        assert target.unmet_prereqs == []

    def test_weak_prereq_does_not_block(self):
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_weak", grade=7),
        ]
        prereqs = {"kc_target": [make_prereq("kc_weak", strength=0.3)]}
        mastery = {"kc_weak": 0.0}
        result = self._run(kcs, prereqs, mastery, grade=8)
        target = next(c for c in result if c.kc_id == "kc_target")
        assert target.unmet_prereqs == []

    def test_multiple_prereqs_all_must_be_met(self):
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_a", grade=7),
            make_kc("kc_b", grade=7),
        ]
        prereqs = {"kc_target": [
            make_prereq("kc_a", strength=0.9),
            make_prereq("kc_b", strength=0.9),
        ]}
        mastery = {"kc_a": 0.8, "kc_b": 0.3}
        result = self._run(kcs, prereqs, mastery, grade=8)
        target = next(c for c in result if c.kc_id == "kc_target")
        assert "kc_b" in target.unmet_prereqs
        assert "kc_a" not in target.unmet_prereqs

    # --- assumed mastery (below grade floor) ---

    def test_prereq_below_grade_floor_assumed_mastered(self):
        """Пререквизит grade=4 для ученика grade=8 (floor=6) → assumed mastered → ready."""
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_old_prereq", grade=4),
        ]
        prereqs = {"kc_target": [make_prereq("kc_old_prereq", strength=0.9)]}
        # mastery не задана — assumed mastered из-за grade < floor
        result = self._run(kcs, prereqs, mastery={}, grade=8)
        target = next(c for c in result if c.kc_id == "kc_target")
        assert target.unmet_prereqs == []

    def test_prereq_at_grade_floor_not_assumed(self):
        """Пререквизит grade=6 для ученика grade=8 (floor=6) → НЕ assumed (grade не < floor)."""
        kcs = [
            make_kc("kc_target", grade=8),
            make_kc("kc_floor_prereq", grade=6),
        ]
        prereqs = {"kc_target": [make_prereq("kc_floor_prereq", strength=0.9)]}
        result = self._run(kcs, prereqs, mastery={}, grade=8)
        target = next(c for c in result if c.kc_id == "kc_target")
        assert "kc_floor_prereq" in target.unmet_prereqs

    # --- sorting ---

    def test_ready_kcs_before_not_ready(self):
        kcs = [
            make_kc("kc_blocked", grade=8, difficulty=0.3),
            make_kc("kc_ready",   grade=8, difficulty=0.8),
            make_kc("kc_missing", grade=7),
        ]
        prereqs = {"kc_blocked": [make_prereq("kc_missing", strength=0.9)]}
        mastery = {"kc_missing": 0.0}
        result = self._run(kcs, prereqs, mastery, grade=8)
        ready = [c for c in result if c.kc_id != "kc_missing"]
        assert ready[0].kc_id == "kc_ready"

    def test_ready_kcs_sorted_by_difficulty(self):
        kcs = [
            make_kc("kc_hard",   grade=8, difficulty=0.9),
            make_kc("kc_easy",   grade=8, difficulty=0.3),
            make_kc("kc_medium", grade=8, difficulty=0.6),
        ]
        result = self._run(kcs, {}, {}, grade=8)
        difficulties = [c.difficulty_base for c in result]
        assert difficulties == sorted(difficulties)

    # --- end-to-end ---

    def test_typical_student_scenario(self):
        """
        Ученик 8 класса. Освоил базу. kc_pythagorean_know должен быть в ZPD.
        Пререквизиты kc_square_power (grade=6) ниже floor → assumed mastered.
        """
        kcs = [
            make_kc("kc_square_power",        grade=6),
            make_kc("kc_sqrt_concept",        grade=7),
            make_kc("kc_right_triangle_parts",grade=7),
            make_kc("kc_pythagorean_know",    grade=8),
        ]
        prereqs = {
            "kc_pythagorean_know": [
                make_prereq("kc_square_power",         strength=0.8),
                make_prereq("kc_sqrt_concept",         strength=0.7),
                make_prereq("kc_right_triangle_parts", strength=0.9),
            ]
        }
        mastery = {
            "kc_square_power":        0.95,  # освоен → вне ZPD
            "kc_sqrt_concept":        0.8,   # освоен
            "kc_right_triangle_parts": 0.75, # освоен
        }
        result = self._run(kcs, prereqs, mastery, grade=8)
        kc_ids = [c.kc_id for c in result]

        assert "kc_square_power" not in kc_ids
        assert "kc_pythagorean_know" in kc_ids
        ready = next(c for c in result if c.kc_id == "kc_pythagorean_know")
        assert ready.unmet_prereqs == []
