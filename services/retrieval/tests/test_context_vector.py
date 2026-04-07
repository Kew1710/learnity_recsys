"""Юнит-тесты _build_context из main.py."""
import math
import numpy as np
import pytest
from services.retrieval.linucb import CONTEXT_DIM
from services.retrieval.main import _build_context


class TestBuildContext:
    def test_shape(self):
        x = _build_context(mastery_kc=0.5, grade=6, avg_reward=0.0, count=0)
        assert x.shape == (CONTEXT_DIM,)

    def test_mastery_kc_at_x0(self):
        x = _build_context(mastery_kc=0.75, grade=5, avg_reward=0.0, count=0)
        assert x[0] == 0.75

    def test_prereqs_fill_x1_to_x4(self):
        prereqs = [0.9, 0.8, 0.7, 0.6]
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                           prereq_masteries=prereqs)
        assert x[1] == 0.9
        assert x[2] == 0.8
        assert x[3] == 0.7
        assert x[4] == 0.6

    def test_prereqs_fewer_than_4_pads_with_zeros(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                           prereq_masteries=[0.8, 0.6])
        assert x[1] == 0.8
        assert x[2] == 0.6
        assert x[3] == 0.0
        assert x[4] == 0.0

    def test_prereqs_more_than_4_truncated(self):
        prereqs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                           prereq_masteries=prereqs)
        assert x[1] == 0.9
        assert x[4] == 0.6
        # x[5] is errors_streak, not x[5th prereq]
        assert x[5] == 0.0

    def test_no_prereqs_defaults_to_zeros(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0)
        assert x[1] == 0.0
        assert x[2] == 0.0
        assert x[3] == 0.0
        assert x[4] == 0.0

    def test_errors_streak_at_x5(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                           errors_streak=3)
        assert x[5] == 3.0

    def test_grade_normalized_at_x6(self):
        x = _build_context(mastery_kc=0.5, grade=11, avg_reward=0.0, count=0)
        assert abs(x[6] - 1.0) < 1e-9

        x = _build_context(mastery_kc=0.5, grade=1, avg_reward=0.0, count=0)
        assert abs(x[6] - 1.0 / 11.0) < 1e-9

    def test_avg_reward_at_x7(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.85, count=0)
        assert x[7] == 0.85

    def test_log_count_at_x8(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=9)
        assert abs(x[8] - math.log(10)) < 1e-9

    def test_count_zero_gives_log_1(self):
        x = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0)
        assert x[8] == 0.0

    def test_none_prereqs_same_as_empty(self):
        x_none = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                                prereq_masteries=None)
        x_empty = _build_context(mastery_kc=0.5, grade=5, avg_reward=0.0, count=0,
                                 prereq_masteries=[])
        assert np.allclose(x_none, x_empty)
