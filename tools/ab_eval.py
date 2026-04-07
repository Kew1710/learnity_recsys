"""
A/B eval: LinUCB (treatment) vs эвристика (control).

Считает session_delta (сумма reward за сессию = 1 час) для каждой группы.
Запускает Mann-Whitney U тест.
Проверяет правило остановки: алерт если одна группа хуже на 30% при 50+ сессиях.

Использование:
    python -m tools.ab_eval
    python -m tools.ab_eval --experiment linucb_v1 --min-sessions 30
"""

import argparse
import asyncio
import sys

import sqlalchemy as sa
from scipy import stats

from shared.db import AsyncSessionLocal

EXPERIMENT_ID = "linucb_v1"
MIN_SESSIONS = 50          # минимум сессий в каждой группе для значимости
STOP_THRESHOLD = 0.30      # алерт если одна группа хуже на 30%
P_VALUE_THRESHOLD = 0.05


async def fetch_session_deltas(experiment_id: str) -> dict[str, list[float]]:
    """
    Возвращает {variant: [session_delta, ...]} из bandit_log + student_experiments.
    Сессия = все взаимодействия одного ученика в рамках одного часа.
    """
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("""
                SELECT
                    e.variant,
                    b.student_id,
                    DATE_TRUNC('hour', b.recommended_at) AS session_hour,
                    SUM(b.reward)                         AS session_delta
                FROM bandit_log b
                JOIN student_experiments e
                    ON e.student_id = b.student_id
                   AND e.experiment_id = :experiment_id
                WHERE b.reward IS NOT NULL
                GROUP BY e.variant, b.student_id, DATE_TRUNC('hour', b.recommended_at)
            """),
            {"experiment_id": experiment_id},
        )).fetchall()

    result: dict[str, list[float]] = {"control": [], "treatment": []}
    for variant, _student_id, _session, delta in rows:
        if variant in result:
            result[variant].append(float(delta))
    return result


def run_analysis(
    deltas: dict[str, list[float]],
    min_sessions: int = MIN_SESSIONS,
) -> dict:
    control = deltas.get("control", [])
    treatment = deltas.get("treatment", [])

    n_ctrl = len(control)
    n_treat = len(treatment)

    print(f"\n{'='*50}")
    print(f"A/B Eval: {EXPERIMENT_ID}")
    print(f"{'='*50}")
    print(f"Control sessions:   {n_ctrl}")
    print(f"Treatment sessions: {n_treat}")

    if n_ctrl == 0 or n_treat == 0:
        print("\n⚠  Недостаточно данных.")
        return {"status": "insufficient_data"}

    avg_ctrl = sum(control) / n_ctrl
    avg_treat = sum(treatment) / n_treat

    print(f"\nAvg session Δmastery:")
    print(f"  Control:   {avg_ctrl:.4f}")
    print(f"  Treatment: {avg_treat:.4f}")

    # Mann-Whitney U (непараметрический)
    u_stat, p_value = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    print(f"\nMann-Whitney U: {u_stat:.1f},  p = {p_value:.4f}")

    significant = p_value < P_VALUE_THRESHOLD
    if significant:
        winner = "treatment" if avg_treat > avg_ctrl else "control"
        print(f"✅ Статистически значимо (p < {P_VALUE_THRESHOLD}). Лидер: {winner}")
    else:
        print(f"⬜ Разница не значима (p ≥ {P_VALUE_THRESHOLD})")

    # Правило остановки
    stop_alert = False
    if n_ctrl >= min_sessions and n_treat >= min_sessions:
        if avg_ctrl > 0 and avg_treat < avg_ctrl * (1 - STOP_THRESHOLD):
            stop_alert = True
            print(f"\n🔴 СТОП: treatment хуже control на >{STOP_THRESHOLD*100:.0f}% при {n_treat} сессиях!")
        elif avg_treat > 0 and avg_ctrl < avg_treat * (1 - STOP_THRESHOLD):
            stop_alert = True
            print(f"\n🔴 СТОП: control хуже treatment на >{STOP_THRESHOLD*100:.0f}% при {n_ctrl} сессиях!")
    else:
        print(f"\nℹ  Правило остановки не применяется (нужно ≥{min_sessions} сессий в каждой группе)")

    print(f"{'='*50}\n")

    return {
        "status": "ok",
        "n_control": n_ctrl,
        "n_treatment": n_treat,
        "avg_control": avg_ctrl,
        "avg_treatment": avg_treat,
        "p_value": p_value,
        "significant": significant,
        "stop_alert": stop_alert,
    }


async def main(experiment_id: str, min_sessions: int) -> int:
    deltas = await fetch_session_deltas(experiment_id)
    result = run_analysis(deltas, min_sessions)
    return 1 if result.get("stop_alert") else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B eval for LinUCB experiment")
    parser.add_argument("--experiment", default=EXPERIMENT_ID)
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.experiment, args.min_sessions))
    sys.exit(exit_code)
