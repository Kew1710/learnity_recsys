"""
Логика выбора KC и задания — чистые функции без HTTP и БД.

select_kc_from_zpd:
  Принимает ZPD-кандидатов, возвращает лучшую KC с учётом subject rotation.
  Приоритет: готовые KC (все пререквизиты освоены) → по сложности.
  Subject rotation: если последнее задание было по алгебре — предпочтём геометрию.

select_task:
  Из списка заданий для KC выбирает одно.
  Exploitation (80%): задание с difficulty ближайшей к ZPD-целевой сложности.
  Stretch exploration (20%): задание с difficulty ближайшей к mastery+0.4 —
    быстрая калибровка BKT для недооценённых учеников.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass

EXPLORATION_RATE = 0.20
TARGET_ZPD_ACCURACY = 0.65   # целевая вероятность правильного ответа в ZPD


def compute_p_correct(mastery: float, irt_difficulty: float) -> float:
    """
    Вероятность правильного ответа по IRT (модель Раша).
    mastery хранится как вероятность знания [0,1], а irt_difficulty уже
    находится в Rasch/logit-шкале.
    """
    m = max(0.01, min(0.99, mastery))
    theta = math.log(m / (1.0 - m))
    return 1.0 / (1.0 + math.exp(-(theta - irt_difficulty)))


def compute_zpd_target_difficulty(mastery: float, difficulty_mode: str = "build") -> float:
    """
    Эвристическая ZPD-цель на той же вероятностной шкале, что и mastery/task bank.
    Для retrieval нам важен относительный сдвиг сложности вокруг текущего mastery:
      build:        чуть выше текущего уровня
      consolidate:  чуть ниже текущего уровня
      test:         заметно выше текущего уровня
    """
    m = max(0.0, min(1.0, mastery))
    offsets = {
        "build": 0.10,
        "consolidate": -0.10,
        "test": 0.30,
    }
    target = m + offsets.get(difficulty_mode, offsets["build"])
    return max(0.05, min(0.95, target))


@dataclass
class ZPDEntry:
    kc_id: str
    subject: str
    difficulty_base: float
    mastery_effective: float
    ready: bool
    plan_priority: float = 0.0   # 0 если KC не в плане, 0..1 если в плане


def select_kc_from_zpd(
    candidates: list[ZPDEntry],
    last_subject: str | None = None,
    rng: random.Random | None = None,
) -> ZPDEntry | None:
    """
    Выбирает KC для следующего задания.

    Сначала берём только готовые KC (все пререквизиты освоены).
    Если готовых нет — берём из всех кандидатов.
    Subject rotation: если есть альтернатива по предмету — предпочитаем её.
    Рандомизация среди топ-3 кандидатов — предотвращает зацикливание на одних KC.
    """
    if not candidates:
        return None

    _rng = rng or random.Random()

    ready = [c for c in candidates if c.ready]
    pool = ready if ready else candidates

    # Plan KC имеют абсолютный приоритет — subject rotation их не отменяет,
    # и они не ограничены пулом ready (учитель сам решает, что ученик должен изучать)
    plan_kcs = sorted(
        [c for c in candidates if c.plan_priority > 0],
        key=lambda c: c.plan_priority,
        reverse=True,
    )
    if plan_kcs:
        return plan_kcs[0]

    # Subject rotation: только если нет KC из плана
    if last_subject and len(pool) > 1:
        rotated = [c for c in pool if c.subject != last_subject]
        if rotated:
            pool = rotated

    return _rng.choice(pool)


def select_task(
    tasks: list[dict],
    target_difficulty: float,
    mastery: float | None = None,
    stretch_difficulty: float | None = None,
    exploration_rate: float = EXPLORATION_RATE,
    rng: random.Random | None = None,
) -> tuple[dict, str]:
    """
    Выбирает задание из списка кандидатов.

    target_difficulty: ZPD-целевая сложность (compute_zpd_target_difficulty).
    stretch_difficulty: сложность для stretch exploration (mastery + 0.4).
      Если задан — exploration даёт задачу выше уровня вместо случайной,
      что ускоряет калибровку BKT для недооценённых учеников.

    Возвращает (task, recommendation_source).
    recommendation_source: "zpd" | "stretch" | "exploration"
    """
    if not tasks:
        raise ValueError("Список заданий пуст")

    _rng = rng or random.Random()

    if len(tasks) == 1 or _rng.random() >= exploration_rate:
        # Exploitation: задание с predicted accuracy ближайшей к TARGET_ZPD_ACCURACY (0.65)
        # Если mastery не передан — fallback на старую логику по target_difficulty
        if mastery is not None:
            best = min(
                tasks,
                key=lambda t: abs(
                    compute_p_correct(mastery, t["parts"][0].get("irt_difficulty") or 0.5)
                    - TARGET_ZPD_ACCURACY
                ),
            )
        else:
            best = min(
                tasks,
                key=lambda t: abs((t["parts"][0].get("irt_difficulty") or 0.5) - target_difficulty),
            )
        return best, "zpd"
    elif stretch_difficulty is not None:
        # Stretch exploration: задача выше уровня — быстрая калибровка BKT
        best = min(
            tasks,
            key=lambda t: abs((t["parts"][0].get("irt_difficulty") or 0.5) - stretch_difficulty),
        )
        return best, "stretch"
    else:
        return _rng.choice(tasks), "exploration"
