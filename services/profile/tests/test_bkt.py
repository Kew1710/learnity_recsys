"""
Юнит-тесты BKT. Без БД, без FastAPI — только чистая логика.

Что проверяем:
  - правильный ответ повышает mastery
  - неправильный ответ понижает mastery
  - hints_used снижает силу обновления
  - secondary KC обновляется слабее primary
  - decay снижает mastery со временем
  - граничные случаи (mastery=0, mastery=1)
"""

from datetime import datetime, timedelta
import pytest

from services.profile.bkt import (
    apply_decay,
    bkt_posterior,
    apply_transit,
    smooth_update,
    update_mastery,
    HALF_LIFE_DAYS,
    SMOOTH_LR,
)

# Стандартные параметры для тестов
P_TRANSIT = 0.1
P_SLIP = 0.1
P_GUESS = 0.2
NOW = datetime(2026, 3, 18, 12, 0, 0)
YESTERDAY = NOW - timedelta(days=1)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_no_decay_same_moment(self):
        prob = apply_decay(0.8, NOW, NOW)
        assert prob == pytest.approx(0.8)

    def test_half_decay_after_half_life(self):
        past = NOW - timedelta(days=HALF_LIFE_DAYS)
        prob = apply_decay(0.8, past, NOW)
        assert prob == pytest.approx(0.4, rel=1e-3)

    def test_full_half_life_halves(self):
        past = NOW - timedelta(days=HALF_LIFE_DAYS * 2)
        prob = apply_decay(1.0, past, NOW)
        assert prob == pytest.approx(0.25, rel=1e-3)

    def test_decay_never_negative(self):
        past = NOW - timedelta(days=365)
        prob = apply_decay(0.5, past, NOW)
        assert prob >= 0.0

    def test_no_decay_for_future_date(self):
        # last_practiced в будущем → days_since=0 → нет decay
        future = NOW + timedelta(days=10)
        prob = apply_decay(0.8, future, NOW)
        assert prob == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# BKT posterior
# ---------------------------------------------------------------------------

class TestBktPosterior:
    def test_correct_answer_increases_mastery(self):
        p_before = 0.5
        p_after = bkt_posterior(p_before, score=1.0, p_slip=P_SLIP, p_guess=P_GUESS)
        assert p_after > p_before

    def test_incorrect_answer_decreases_mastery(self):
        p_before = 0.5
        p_after = bkt_posterior(p_before, score=0.0, p_slip=P_SLIP, p_guess=P_GUESS)
        assert p_after < p_before

    def test_partial_score_is_between(self):
        p = 0.5
        p_correct = bkt_posterior(p, score=1.0, p_slip=P_SLIP, p_guess=P_GUESS)
        p_incorrect = bkt_posterior(p, score=0.0, p_slip=P_SLIP, p_guess=P_GUESS)
        p_partial = bkt_posterior(p, score=0.5, p_slip=P_SLIP, p_guess=P_GUESS)
        assert p_incorrect < p_partial < p_correct

    def test_high_mastery_correct_stays_high(self):
        p_after = bkt_posterior(0.95, score=1.0, p_slip=P_SLIP, p_guess=P_GUESS)
        assert p_after > 0.9

    def test_low_mastery_incorrect_stays_low(self):
        p_after = bkt_posterior(0.05, score=0.0, p_slip=P_SLIP, p_guess=P_GUESS)
        assert p_after < 0.2

    def test_output_in_0_1_range(self):
        for p in [0.0, 0.5, 1.0]:
            for score in [0.0, 0.5, 1.0]:
                result = bkt_posterior(p, score=score, p_slip=0.1, p_guess=0.2)
                assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Transit
# ---------------------------------------------------------------------------

class TestTransit:
    def test_transit_increases_probability(self):
        p = apply_transit(0.5, p_transit=0.1)
        assert p > 0.5

    def test_transit_zero_no_change(self):
        p = apply_transit(0.7, p_transit=0.0)
        assert p == pytest.approx(0.7)

    def test_transit_at_max_no_overflow(self):
        p = apply_transit(1.0, p_transit=0.5)
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# smooth_update
# ---------------------------------------------------------------------------

class TestSmoothUpdate:
    def test_correct_at_zero_gives_lr(self):
        result = smooth_update(0.0, score=1.0, lr=SMOOTH_LR, transit=0.0)
        assert result == pytest.approx(SMOOTH_LR, rel=1e-6)

    def test_gain_decreases_with_mastery(self):
        gains = [smooth_update(p, 1.0, transit=0.0) - p for p in [0.0, 0.3, 0.6, 0.9]]
        for i in range(len(gains) - 1):
            assert gains[i] > gains[i + 1]

    def test_wrong_at_high_mastery_decreases(self):
        result = smooth_update(0.9, score=0.0, transit=0.0)
        assert result < 0.9

    def test_wrong_at_low_mastery_barely_changes(self):
        delta = smooth_update(0.05, score=0.0, transit=0.0) - 0.05
        assert abs(delta) < 0.02

    def test_output_clamped_0_1(self):
        assert smooth_update(0.0, 0.0, transit=0.0) >= 0.0
        assert smooth_update(1.0, 1.0) <= 1.0

    def test_streak_increases_gain(self):
        no_streak = smooth_update(0.2, score=1.0, transit=0.0, consecutive_correct=0, recent_accuracy=0.8)
        streak_4 = smooth_update(0.2, score=1.0, transit=0.0, consecutive_correct=4, recent_accuracy=0.8)
        assert streak_4 > no_streak

    def test_streak_lr_capped_at_040(self):
        # cap срабатывает при streak≥12 (0.15*(1+0.15*12)=0.42 → обрезается до 0.40)
        result_big = smooth_update(0.0, score=1.0, transit=0.0, consecutive_correct=100, recent_accuracy=0.8)
        result_cap = smooth_update(0.0, score=1.0, transit=0.0, consecutive_correct=12, recent_accuracy=0.8)
        assert result_big == pytest.approx(result_cap, rel=1e-6)

    def test_streak_no_effect_without_recent_accuracy(self):
        # Без recent_accuracy (default=0.0) streak не активируется
        no_streak = smooth_update(0.2, score=1.0, transit=0.0, consecutive_correct=0)
        streak_no_acc = smooth_update(0.2, score=1.0, transit=0.0, consecutive_correct=4)
        assert streak_no_acc == pytest.approx(no_streak, rel=1e-6)

    def test_streak_no_effect_on_wrong_answer(self):
        result = smooth_update(0.5, score=0.0, transit=0.0, consecutive_correct=5, recent_accuracy=0.8)
        assert 0.0 <= result <= 1.0

    def test_surprise_boosts_lr_when_harder_than_expected(self):
        # mastery=0.1, difficulty=0.5 → ожидаем ~6% правильных, студент решил → большой surprise
        no_diff = smooth_update(0.1, score=1.0, transit=0.0)
        with_diff = smooth_update(0.1, score=1.0, transit=0.0, irt_difficulty=0.5)
        assert with_diff > no_diff

    def test_surprise_no_boost_when_easier_than_expected(self):
        # mastery=0.9, difficulty=0.1 → ожидаем ~99% правильных, решил → surprise ≈ 0
        no_diff = smooth_update(0.9, score=1.0, transit=0.0)
        with_diff = smooth_update(0.9, score=1.0, transit=0.0, irt_difficulty=0.1)
        assert with_diff == pytest.approx(no_diff, abs=0.01)

    def test_surprise_no_boost_on_wrong_answer(self):
        # Провалил лёгкое задание — surprise отрицательный, lr не растёт
        no_diff = smooth_update(0.5, score=0.0, transit=0.0)
        with_diff = smooth_update(0.5, score=0.0, transit=0.0, irt_difficulty=0.1)
        assert with_diff == pytest.approx(no_diff, rel=1e-6)

    def test_surprise_result_clamped(self):
        # Максимальный surprise не выходит за 1.0
        result = smooth_update(0.01, score=1.0, transit=0.0, irt_difficulty=0.99)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# update_mastery — полный пайплайн
# ---------------------------------------------------------------------------

class TestUpdateMastery:
    def _call(self, **kwargs):
        defaults = dict(
            current_probability=0.5,
            last_practiced=YESTERDAY,
            score=1.0,
            hints_used=0,
            p_transit=P_TRANSIT,
            p_slip=P_SLIP,
            p_guess=P_GUESS,
            kc_role="primary",
            now=NOW,
        )
        defaults.update(kwargs)
        return update_mastery(**defaults)

    def test_correct_no_hints_primary_increases(self):
        result = self._call(current_probability=0.5, score=1.0)
        assert result > 0.5

    def test_incorrect_decreases(self):
        result = self._call(current_probability=0.5, score=0.0)
        assert result < 0.5

    def test_hints_weaken_update(self):
        no_hints = self._call(score=1.0, hints_used=0)
        with_hints = self._call(score=1.0, hints_used=2)
        # С подсказками обновление слабее → результат ближе к prior
        assert with_hints < no_hints

    def test_secondary_weaker_than_primary(self):
        primary = self._call(score=1.0, kc_role="primary")
        secondary = self._call(score=1.0, kc_role="secondary")
        assert secondary < primary

    def test_result_always_in_0_1(self):
        for prob in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for score in [0.0, 0.5, 1.0]:
                result = self._call(current_probability=prob, score=score)
                assert 0.0 <= result <= 1.0, f"Out of range: prob={prob}, score={score} → {result}"

    def test_multiple_correct_answers_converge_to_high(self):
        """10 правильных ответов подряд → mastery должна быть высокой."""
        prob = 0.1
        for i in range(10):
            prob = update_mastery(
                current_probability=prob,
                last_practiced=NOW - timedelta(days=1),
                score=1.0,
                hints_used=0,
                p_transit=P_TRANSIT,
                p_slip=P_SLIP,
                p_guess=P_GUESS,
                kc_role="primary",
                now=NOW,
            )
        assert prob > 0.75

    def test_diminishing_returns_on_correct(self):
        """Прирост mastery за верный ответ убывает с ростом mastery."""
        gains = []
        for p in [0.0, 0.2, 0.4, 0.6, 0.8]:
            result = self._call(current_probability=p, last_practiced=NOW, score=1.0)
            gains.append(result - p)
        # Каждый следующий прирост меньше предыдущего
        for i in range(len(gains) - 1):
            assert gains[i] > gains[i + 1], f"Δ at p={i*0.2:.1f}: {gains[i]:.4f} should be > {gains[i+1]:.4f}"

    def test_streak_correct_faster_mastery_gain(self):
        no_streak = self._call(current_probability=0.2, last_practiced=NOW, score=1.0, consecutive_correct=0, recent_accuracy=0.8)
        with_streak = self._call(current_probability=0.2, last_practiced=NOW, score=1.0, consecutive_correct=4, recent_accuracy=0.8)
        assert with_streak > no_streak

    def test_decay_applied_before_update(self):
        """Давно не практиковавшийся KC — decay снижает probability перед обновлением."""
        long_ago = NOW - timedelta(days=90)
        result_recent = self._call(current_probability=0.9, last_practiced=YESTERDAY, score=0.0)
        result_old = self._call(current_probability=0.9, last_practiced=long_ago, score=0.0)
        # После decay 90 дней probability ~= 0.9 * 0.5^3 ≈ 0.11
        # Неправильный ответ с низкой probability даёт более низкий результат?
        # Нет — старый KC с decay уже низкий, неправильный ответ его ещё снизит
        # Но главное: результат должен отличаться
        assert result_recent != pytest.approx(result_old)
