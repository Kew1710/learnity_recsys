"""
TasksToMasteryEstimator — оценивает сколько задач нужно для освоения KC.

simulate_tasks_to_mastery:
  Запускает BKT симуляцию N раз, возвращает медиану.

estimate:
  Гибридная оценка:
    α = min(1.0, n_practiced_in_subject / 20)  — вес личной истории
    result = α × cluster_avg + (1-α) × simulation_estimate
"""

from __future__ import annotations

import random
import statistics

from services.profile.bkt import smooth_update, SMOOTH_LR, SMOOTH_TRANSIT

DEFAULT_P_TRANSIT = SMOOTH_LR   # lr в EMA (= 0.22, соответствует реальной системе)
DEFAULT_P_SLIP = 0.10           # не используется, сохранён для обратной совместимости API
DEFAULT_P_GUESS = 0.20          # не используется, сохранён для обратной совместимости API
DEFAULT_P_CORRECT = 0.70        # ожидаемая точность студента при ZPD-подборе задач
MAX_STEPS = 300                 # защита от бесконечного цикла в симуляции


def simulate_tasks_to_mastery(
    m_current: float,
    m_target: float,
    p_transit: float = DEFAULT_P_TRANSIT,
    p_slip: float = DEFAULT_P_SLIP,
    p_guess: float = DEFAULT_P_GUESS,
    n_sims: int = 500,
    rng: random.Random | None = None,
) -> int:
    """
    Монте-Карло симуляция: сколько задач нужно чтобы mastery достигла m_target.
    Возвращает медиану по n_sims запускам.

    Использует EMA (smooth_update) как в реальной системе.
    p_transit интерпретируется как lr (скорость обучения).
    p_slip/p_guess сохранены для совместимости API, но не используются.
    """
    if m_current >= m_target:
        return 0

    _rng = rng or random.Random()
    counts: list[int] = []

    for _ in range(n_sims):
        m = m_current
        steps = 0
        while m < m_target and steps < MAX_STEPS:
            score = 1.0 if _rng.random() < DEFAULT_P_CORRECT else 0.0
            m = smooth_update(m, score, lr=p_transit, transit=SMOOTH_TRANSIT)
            steps += 1
        counts.append(steps)

    return int(statistics.median(counts))


def estimate(
    m_current: float,
    m_target: float,
    cluster_avg: float | None,
    n_practiced_in_subject: int,
    p_transit: float = DEFAULT_P_TRANSIT,
    p_slip: float = DEFAULT_P_SLIP,
    p_guess: float = DEFAULT_P_GUESS,
    n_sims: int = 500,
    rng: random.Random | None = None,
) -> int:
    """
    Гибридная оценка tasks_to_mastery.

    Args:
        m_current: текущий mastery ученика для KC
        m_target: целевой mastery
        cluster_avg: средняя оценка tasks_to_mastery по кластеру (None если нет данных)
        n_practiced_in_subject: количество задач решённых учеником по этому предмету
        p_transit/slip/guess: BKT параметры

    Returns:
        оценка числа задач (целое)
    """
    sim = simulate_tasks_to_mastery(
        m_current, m_target, p_transit, p_slip, p_guess, n_sims, rng
    )

    if cluster_avg is None:
        return sim

    alpha = min(1.0, n_practiced_in_subject / 20.0)
    blended = alpha * cluster_avg + (1.0 - alpha) * sim
    return max(1, int(round(blended)))
