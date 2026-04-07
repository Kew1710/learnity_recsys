"""Тесты coverage scoring (policy_mode2) и OnBudgetAlert."""
import pytest
from services.macro.policy_mode2 import (
    score_kc,
    rank_coverage_kcs,
    MASTERY_THRESHOLD,
    PREREQ_READY_MIN,
)


# ---------------------------------------------------------------------------
# score_kc — фильтры
# ---------------------------------------------------------------------------

class TestScoreKcFilters:
    def test_grade_above_student_returns_minus_one(self):
        assert score_kc("kc_x", {}, [], grade_introduced=10, student_grade=7) == -1.0

    def test_grade_equal_to_student_is_allowed(self):
        s = score_kc("kc_x", {"kc_x": 0.3}, [], grade_introduced=7, student_grade=7)
        assert s > 0

    def test_mastered_kc_returns_minus_one(self):
        assert score_kc("kc_x", {"kc_x": 0.96}, [], grade_introduced=5, student_grade=8) == -1.0

    def test_zero_mastery_not_filtered(self):
        s = score_kc("kc_x", {}, [], grade_introduced=5, student_grade=8)
        assert s > 0


# ---------------------------------------------------------------------------
# score_kc — варианты
# ---------------------------------------------------------------------------

class TestScoreKcVariants:
    BASE = dict(grade_introduced=7, student_grade=8, prereq_masteries=[], budget_ratio=1.0)

    def test_count_higher_for_near_threshold(self):
        """count: KC близкая к порогу (0.70) > KC далёкая (0.20)."""
        near = score_kc("a", {"a": 0.70}, variant="count", **self.BASE)
        far  = score_kc("b", {"b": 0.20}, variant="count", **self.BASE)
        assert near > far

    def test_mass_higher_for_low_mastery(self):
        """mass: KC с mastery=0.0 > mastery=0.70 (больше потенциальный прирост)."""
        low  = score_kc("a", {"a": 0.00}, variant="mass", **self.BASE)
        high = score_kc("b", {"b": 0.70}, variant="mass", **self.BASE)
        assert low > high

    def test_frontier_bonus_for_untouched_kc(self):
        """frontier: KC с mastery<0.05 получает x2 бонус."""
        untouched = score_kc("a", {"a": 0.00}, variant="frontier", **self.BASE)
        touched   = score_kc("b", {"b": 0.40}, variant="frontier", **self.BASE)
        # При одинаковом gap untouched должен быть выше
        untouched_no_bonus = score_kc("a", {"a": 0.00}, variant="count", **self.BASE)
        assert untouched > untouched_no_bonus  # frontier даёт бонус

    def test_all_variants_return_nonnegative(self):
        for variant in ("count", "mass", "frontier"):
            s = score_kc("kc_x", {"kc_x": 0.3}, [], 6, 8, variant=variant)
            assert s >= 0


# ---------------------------------------------------------------------------
# score_kc — prereq_weight
# ---------------------------------------------------------------------------

class TestScoreKcPrereqs:
    def test_weak_prereqs_reduce_score(self):
        no_prereqs  = score_kc("kc_x", {"kc_x": 0.3}, prereq_masteries=[], grade_introduced=6, student_grade=8)
        weak_prereqs = score_kc("kc_x", {"kc_x": 0.3}, prereq_masteries=[0.2, 0.1], grade_introduced=6, student_grade=8)
        assert weak_prereqs < no_prereqs

    def test_strong_prereqs_dont_penalize(self):
        no_prereqs    = score_kc("kc_x", {"kc_x": 0.3}, prereq_masteries=[], grade_introduced=6, student_grade=8)
        strong_prereqs = score_kc("kc_x", {"kc_x": 0.3}, prereq_masteries=[0.9, 0.85], grade_introduced=6, student_grade=8)
        assert abs(no_prereqs - strong_prereqs) < 1e-9  # prereq_weight=1.0 в обоих случаях

    def test_very_weak_prereqs_still_positive(self):
        """KC с очень слабыми prereqs — низкий балл, но не -1."""
        s = score_kc("kc_x", {"kc_x": 0.3}, prereq_masteries=[0.1], grade_introduced=6, student_grade=8)
        assert 0 < s


# ---------------------------------------------------------------------------
# score_kc — budget_ratio
# ---------------------------------------------------------------------------

class TestScoreKcBudget:
    def test_low_budget_ignores_frontier_bonus(self):
        """Мало бюджета: frontier-KC и count-KC получают одинаковый score (только gap)."""
        frontier = score_kc("a", {"a": 0.00}, [], 6, 8, variant="frontier", budget_ratio=0.10)
        count    = score_kc("a", {"a": 0.00}, [], 6, 8, variant="count",    budget_ratio=0.10)
        assert frontier == pytest.approx(count)

    def test_full_budget_frontier_above_count(self):
        frontier = score_kc("a", {"a": 0.00}, [], 6, 8, variant="frontier", budget_ratio=1.0)
        count    = score_kc("a", {"a": 0.00}, [], 6, 8, variant="count",    budget_ratio=1.0)
        assert frontier > count


# ---------------------------------------------------------------------------
# rank_coverage_kcs
# ---------------------------------------------------------------------------

class TestRankCoverageKcs:
    def _make_kcs(self, specs):
        """specs: [(kc_id, grade, mastery, prereq_masteries)]"""
        return [
            {"kc_id": kc_id, "grade_introduced": g, "prereq_masteries": pm}
            for kc_id, g, _, pm in specs
        ], {kc_id: m for kc_id, _, m, _ in specs}

    def test_returns_at_most_top_n(self):
        kcs = [{"kc_id": f"kc_{i}", "grade_introduced": 7, "prereq_masteries": []} for i in range(20)]
        mastery = {f"kc_{i}": 0.3 for i in range(20)}
        result = rank_coverage_kcs(kcs, mastery, student_grade=8, top_n=5)
        assert len(result) <= 5

    def test_excludes_out_of_grade(self):
        kcs = [
            {"kc_id": "kc_low",  "grade_introduced": 6, "prereq_masteries": []},
            {"kc_id": "kc_high", "grade_introduced": 10, "prereq_masteries": []},
        ]
        mastery = {"kc_low": 0.3, "kc_high": 0.3}
        result = rank_coverage_kcs(kcs, mastery, student_grade=8)
        assert "kc_high" not in result
        assert "kc_low" in result

    def test_excludes_mastered(self):
        kcs = [
            {"kc_id": "kc_done", "grade_introduced": 7, "prereq_masteries": []},
            {"kc_id": "kc_todo", "grade_introduced": 7, "prereq_masteries": []},
        ]
        mastery = {"kc_done": 0.97, "kc_todo": 0.3}
        result = rank_coverage_kcs(kcs, mastery, student_grade=8)
        assert "kc_done" not in result
        assert "kc_todo" in result

    def test_count_variant_near_threshold_first(self):
        """count: KC с mastery=0.70 должна идти раньше KC с mastery=0.10."""
        kcs = [
            {"kc_id": "near", "grade_introduced": 7, "prereq_masteries": []},
            {"kc_id": "far",  "grade_introduced": 7, "prereq_masteries": []},
        ]
        mastery = {"near": 0.70, "far": 0.10}
        result = rank_coverage_kcs(kcs, mastery, student_grade=8, variant="count")
        assert result.index("near") < result.index("far")

    def test_mass_variant_low_mastery_first(self):
        """mass: KC с mastery=0.0 должна идти раньше KC с mastery=0.60."""
        kcs = [
            {"kc_id": "low",  "grade_introduced": 7, "prereq_masteries": []},
            {"kc_id": "mid",  "grade_introduced": 7, "prereq_masteries": []},
        ]
        mastery = {"low": 0.0, "mid": 0.60}
        result = rank_coverage_kcs(kcs, mastery, student_grade=8, variant="mass")
        assert result.index("low") < result.index("mid")

    def test_empty_result_when_all_mastered(self):
        kcs = [{"kc_id": "kc_x", "grade_introduced": 7, "prereq_masteries": []}]
        result = rank_coverage_kcs(kcs, {"kc_x": 0.98}, student_grade=8)
        assert result == []


# ---------------------------------------------------------------------------
# OnBudgetAlert в evaluate_micro_summary
# ---------------------------------------------------------------------------

class TestOnBudgetAlert:
    def test_budget_alert_triggers_test_mode(self):
        from services.macro.plan_lifecycle import evaluate_micro_summary
        summary = {
            "kc_id": "kc_x", "avg_score": 0.7, "velocity": 0.02,
            "frustration_count": 0, "tasks_spent": 85,
        }
        actions = evaluate_micro_summary(summary, tasks_budget=100)
        assert len(actions) == 1
        assert actions[0].action_type == "set_difficulty_mode"
        assert actions[0].payload["difficulty_mode"] == "test"

    def test_no_budget_alert_if_enough_remaining(self):
        from services.macro.plan_lifecycle import evaluate_micro_summary
        summary = {
            "kc_id": "kc_x", "avg_score": 0.7, "velocity": 0.02,
            "frustration_count": 0, "tasks_spent": 70,
        }
        actions = evaluate_micro_summary(summary, tasks_budget=100)
        assert all(
            a.action_type != "set_difficulty_mode" or a.payload.get("difficulty_mode") != "test"
            for a in actions
        )

    def test_no_alert_without_budget(self):
        from services.macro.plan_lifecycle import evaluate_micro_summary
        summary = {
            "kc_id": "kc_x", "avg_score": 0.7, "velocity": 0.02,
            "frustration_count": 0, "tasks_spent": 95,
        }
        actions = evaluate_micro_summary(summary, tasks_budget=None)
        assert all(a.action_type != "set_difficulty_mode" for a in actions)
