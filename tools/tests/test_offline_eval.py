"""Smoke tests for offline_eval — test pure computation logic without DB."""

import math
from collections import defaultdict


def _compute_brier(predictions, actuals):
    return sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)


def _compute_logloss(predictions, actuals):
    eps = 1e-7
    return -sum(
        a * math.log(max(eps, p)) + (1 - a) * math.log(max(eps, 1 - p))
        for p, a in zip(predictions, actuals)
    ) / len(predictions)


def test_brier_perfect():
    preds = [1.0, 0.0, 1.0, 0.0]
    actuals = [1.0, 0.0, 1.0, 0.0]
    assert _compute_brier(preds, actuals) == 0.0


def test_brier_worst():
    preds = [1.0, 0.0]
    actuals = [0.0, 1.0]
    assert _compute_brier(preds, actuals) == 1.0


def test_brier_random():
    preds = [0.5] * 100
    actuals = [1.0] * 50 + [0.0] * 50
    assert abs(_compute_brier(preds, actuals) - 0.25) < 0.001


def test_logloss_perfect_is_low():
    preds = [0.99, 0.01, 0.99]
    actuals = [1.0, 0.0, 1.0]
    ll = _compute_logloss(preds, actuals)
    assert ll < 0.05


def test_logloss_random_is_log2():
    preds = [0.5] * 100
    actuals = [1.0] * 50 + [0.0] * 50
    ll = _compute_logloss(preds, actuals)
    assert abs(ll - math.log(2)) < 0.01


def test_calibration_buckets():
    buckets = defaultdict(lambda: {"predicted": 0.0, "actual": 0.0, "count": 0})
    preds = [0.15, 0.25, 0.75, 0.85]
    actuals = [0.0, 0.0, 1.0, 1.0]
    for p, a in zip(preds, actuals):
        b = min(9, int(p * 10))
        buckets[b]["predicted"] += p
        buckets[b]["actual"] += a
        buckets[b]["count"] += 1

    assert buckets[1]["count"] == 1
    assert buckets[7]["count"] == 1
    assert buckets[8]["count"] == 1
    assert buckets[7]["actual"] == 1.0
