"""
Offline evaluation pipeline for Knowledge Tracing and Macro Policy.

Evaluates:
  1. Mastery prediction quality (EMA baseline):
     - AUC: can current mastery predict next-task correctness?
     - Calibration: does mastery=0.7 really mean 70% correct?
     - Brier score

  2. Macro policy quality (from macro_transitions):
     - Action distribution: how often each action is taken
     - Diagnosis accuracy: does diagnosed reason correlate with outcome?
     - Average reward per action type

Usage:
    python -m tools.offline_eval
    python -m tools.offline_eval --kt-only
    python -m tools.offline_eval --macro-only
"""

import argparse
import asyncio
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text


async def _eval_kt():
    """Evaluate mastery prediction quality against actual outcomes."""
    from shared.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT
                bl.kc_id,
                bl.context_vector[1] AS mastery_at_rec,
                bl.reward,
                bl.recommended_at
            FROM bandit_log bl
            WHERE bl.reward IS NOT NULL
              AND bl.context_vector IS NOT NULL
              AND array_length(bl.context_vector, 1) >= 1
            ORDER BY bl.recommended_at
        """))).fetchall()

    if not rows:
        print("KT eval: no bandit_log data with rewards found.")
        return

    predictions = []
    actuals = []

    for kc_id, mastery, reward, rec_at in rows:
        if mastery is None:
            continue
        p = max(0.01, min(0.99, float(mastery)))
        actual = 1.0 if float(reward) > 0 else 0.0
        predictions.append(p)
        actuals.append(actual)

    n = len(predictions)
    if n < 10:
        print(f"KT eval: only {n} samples, need at least 10.")
        return

    # Brier score
    brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n

    # Log-loss
    eps = 1e-7
    logloss = -sum(
        a * math.log(max(eps, p)) + (1 - a) * math.log(max(eps, 1 - p))
        for p, a in zip(predictions, actuals)
    ) / n

    # Calibration buckets
    buckets = defaultdict(lambda: {"predicted": 0.0, "actual": 0.0, "count": 0})
    for p, a in zip(predictions, actuals):
        bucket = min(9, int(p * 10))
        buckets[bucket]["predicted"] += p
        buckets[bucket]["actual"] += a
        buckets[bucket]["count"] += 1

    # AUC (simple trapezoidal)
    sorted_pairs = sorted(zip(predictions, actuals), reverse=True)
    tp, fp = 0, 0
    total_pos = sum(actuals)
    total_neg = n - total_pos
    auc_points = []
    for p, a in sorted_pairs:
        if a == 1.0:
            tp += 1
        else:
            fp += 1
        if total_pos > 0 and total_neg > 0:
            auc_points.append((fp / total_neg, tp / total_pos))

    auc = 0.0
    if len(auc_points) > 1:
        for i in range(1, len(auc_points)):
            dx = auc_points[i][0] - auc_points[i - 1][0]
            avg_y = (auc_points[i][1] + auc_points[i - 1][1]) / 2
            auc += dx * avg_y

    print(f"\n=== Knowledge Tracing Evaluation ({n} samples) ===")
    print(f"  Brier Score:  {brier:.4f}  (lower is better, 0.25 = random)")
    print(f"  Log-Loss:     {logloss:.4f}")
    print(f"  AUC:          {auc:.4f}  (0.5 = random, 1.0 = perfect)")
    print()
    print("  Calibration (predicted vs actual success rate):")
    print(f"  {'Bucket':>8} {'Pred':>8} {'Actual':>8} {'Count':>8}")
    for b in sorted(buckets.keys()):
        d = buckets[b]
        avg_pred = d["predicted"] / d["count"]
        avg_actual = d["actual"] / d["count"]
        print(f"  {b * 10:>5}-{(b + 1) * 10:>2}% {avg_pred:>8.3f} {avg_actual:>8.3f} {d['count']:>8}")


async def _eval_macro():
    """Evaluate macro policy quality from logged transitions."""
    from shared.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT
                action,
                reward,
                done,
                reason,
                diagnosis_reason,
                diagnosis_confidence
            FROM macro_transitions
            ORDER BY created_at
        """))).fetchall()

    if not rows:
        print("\nMacro eval: no transitions found in macro_transitions table.")
        return

    action_stats = defaultdict(lambda: {"count": 0, "rewards": [], "done_count": 0})
    diag_stats = defaultdict(lambda: {"count": 0, "actions": defaultdict(int)})

    for action, reward, done, reason, diag_reason, diag_conf in rows:
        action_stats[action]["count"] += 1
        if reward is not None:
            action_stats[action]["rewards"].append(float(reward))
        if done:
            action_stats[action]["done_count"] += 1
        if diag_reason:
            diag_stats[diag_reason]["count"] += 1
            diag_stats[diag_reason]["actions"][action] += 1

    total = len(rows)
    print(f"\n=== Macro Policy Evaluation ({total} transitions) ===")
    print()
    print("  Action distribution:")
    print(f"  {'Action':>25} {'Count':>8} {'%':>8} {'Avg Reward':>12} {'Done %':>8}")
    for action in sorted(action_stats.keys()):
        s = action_stats[action]
        avg_r = sum(s["rewards"]) / len(s["rewards"]) if s["rewards"] else float("nan")
        done_pct = s["done_count"] / s["count"] * 100 if s["count"] > 0 else 0
        print(f"  {action:>25} {s['count']:>8} {s['count'] / total * 100:>7.1f}% {avg_r:>12.4f} {done_pct:>7.1f}%")

    if diag_stats:
        print()
        print("  Diagnosis distribution:")
        print(f"  {'Diagnosis':>25} {'Count':>8} {'Top Actions':>40}")
        for diag in sorted(diag_stats.keys()):
            d = diag_stats[diag]
            top = sorted(d["actions"].items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{a}({c})" for a, c in top)
            print(f"  {diag:>25} {d['count']:>8} {top_str:>40}")


async def main():
    parser = argparse.ArgumentParser(description="Offline evaluation pipeline")
    parser.add_argument("--kt-only", action="store_true", help="Only evaluate KT")
    parser.add_argument("--macro-only", action="store_true", help="Only evaluate macro policy")
    args = parser.parse_args()

    if args.kt_only:
        await _eval_kt()
    elif args.macro_only:
        await _eval_macro()
    else:
        await _eval_kt()
        await _eval_macro()


if __name__ == "__main__":
    asyncio.run(main())
