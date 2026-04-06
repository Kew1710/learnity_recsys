"""
Bayesian Knowledge Tracing (BKT) — алгоритм обновления знаний.

Четыре параметра на KC:
  P(L0) — начальная вероятность знания (хранится в mastery.probability)
  P(T)  — вероятность выучить KC за одну попытку       (p_transit)
  P(S)  — вероятность ошибиться, зная KC               (p_slip)
  P(G)  — вероятность угадать, не зная KC              (p_guess)

Порядок обновления на каждый ответ:
  1. decay(mastery)         — ослабить знание с учётом времени
  2. bkt_posterior(...)     — байесовское обновление по ответу
  3. transit(posterior)     — учесть вероятность выучить в ходе попытки
"""

from __future__ import annotations
import math
from datetime import datetime

HALF_LIFE_DAYS = 30.0   # через 30 дней без практики mastery падает вдвое

PERFORMANCE_DECAY_THRESHOLD = 3    # порог consecutive_errors для активного снижения mastery
PERFORMANCE_DECAY_FACTOR = 0.75    # множитель при систематических ошибках

# Параметры плавного обновления mastery (smooth_update)
SMOOTH_LR = 0.15       # шаг обновления: Δ = lr * (score - p)  →  при p=0: +0.15, при p=0.6: +0.06
SMOOTH_TRANSIT = 0.02  # маленький бонус за сам факт попытки (effort bonus)
SURPRISE_K = 1.0       # коэффициент усиления lr за неожиданный результат (студент решил сложнее чем ожидалось)


# ---------------------------------------------------------------------------
# 1. Decay
# ---------------------------------------------------------------------------

def apply_decay(
    probability: float,
    last_practiced: datetime,
    now: datetime | None = None,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Ослабить хранимую вероятность знания с учётом времени без практики."""
    if now is None:
        now = datetime.utcnow()
    days_since = max(0.0, (now - last_practiced).total_seconds() / 86400)
    decay_factor = 0.5 ** (days_since / half_life_days)
    return probability * decay_factor


# ---------------------------------------------------------------------------
# 2. Smooth mastery update (EMA-based)
# ---------------------------------------------------------------------------

def smooth_update(
    p_mastery: float,
    score: float,
    lr: float = SMOOTH_LR,
    transit: float = SMOOTH_TRANSIT,
    consecutive_correct: int = 0,
    recent_accuracy: float = 0.0,
    irt_difficulty: float | None = None,
) -> float:
    """
    Плавное обновление mastery с убывающей скоростью.

    Формула: new_p = p + effective_lr * (score - p)
    Эквивалентно EMA: new_p = (1-lr)*p + lr*score

    consecutive_correct масштабирует lr вверх — streak правильных ответов
    ускоряет калибровку для недооценённых (быстрых) учеников:
      streak=0: lr=0.15, streak=2: lr≈0.225, streak=4: lr≈0.30, streak=6+: lr=0.40

    irt_difficulty усиливает lr когда студент решил сложнее чем ожидалось (surprise > 0):
      mastery=0.1, difficulty=0.5, score=1.0 → expected≈0.06, surprise≈0.94 → lr×1.94
      mastery=0.5, difficulty=0.5, score=1.0 → expected=0.50, surprise=0.50 → lr×1.50
      mastery=0.5, difficulty=0.5, score=0.0 → surprise<0 → lr без изменений

    Args:
        p_mastery: текущая вероятность знания [0, 1]
        score: результат попытки [0.0–1.0]
        lr: базовая скорость обучения (default: 0.15)
        transit: бонус за попытку (default: 0.02)
        consecutive_correct: количество правильных ответов подряд ДО этой попытки
        irt_difficulty: сложность задания по IRT [0, 1] (опционально)
    """
    # Streak-бонус применяется только если недавняя точность высокая (>= 0.6).
    # Без этой проверки угаданные ответы (p_guess) разгоняют mastery у слабых учеников.
    streak_active = consecutive_correct > 0 and recent_accuracy >= 0.6
    effective_lr = min(0.22, lr * (1.0 + 0.08 * consecutive_correct)) if streak_active and irt_difficulty >= mastery - 0.1 else lr

    # Surprise-бонус: усиливаем lr если студент решил сложнее чем предсказывала модель.
    if irt_difficulty is not None:
        m = max(0.01, min(0.99, p_mastery))
        theta = math.log(m / (1.0 - m))
        expected = 1.0 / (1.0 + math.exp(-(theta - irt_difficulty)))
        surprise = score - expected
        effective_lr = min(0.22, effective_lr * (1.0 + SURPRISE_K * max(0.0, surprise)))

    new_p = p_mastery + effective_lr * (score - p_mastery)
    new_p = new_p + transit * (1.0 - new_p)
    return max(0.0, min(1.0, new_p))


# ---------------------------------------------------------------------------
# 3. BKT posterior (байесовское обновление) — используется в RL-симуляторе
# ---------------------------------------------------------------------------

def _bkt_correct(p_mastery: float, p_slip: float, p_guess: float) -> float:
    """P(mastered | correct response)"""
    numerator = p_mastery * (1 - p_slip)
    denominator = numerator + (1 - p_mastery) * p_guess
    if denominator == 0:
        return p_mastery
    return numerator / denominator


def _bkt_incorrect(p_mastery: float, p_slip: float, p_guess: float) -> float:
    """P(mastered | incorrect response)"""
    numerator = p_mastery * p_slip
    denominator = numerator + (1 - p_mastery) * (1 - p_guess)
    if denominator == 0:
        return p_mastery
    return numerator / denominator


def bkt_posterior(
    p_mastery: float,
    score: float,       # 0.0–1.0 (непрерывный ответ)
    p_slip: float,
    p_guess: float,
) -> float:
    """
    Интерполируем между p_correct и p_incorrect через score.
    score=1.0 → полностью правильный ответ
    score=0.0 → полностью неправильный
    """
    p_if_correct = _bkt_correct(p_mastery, p_slip, p_guess)
    p_if_incorrect = _bkt_incorrect(p_mastery, p_slip, p_guess)
    return score * p_if_correct + (1 - score) * p_if_incorrect


# ---------------------------------------------------------------------------
# 4. Transit (используется в RL-симуляторе совместно с bkt_posterior)
# ---------------------------------------------------------------------------

def apply_transit(p_posterior: float, p_transit: float) -> float:
    """P(mastered after attempt) = posterior + (1 - posterior) * P(T)"""
    return p_posterior + (1 - p_posterior) * p_transit


# ---------------------------------------------------------------------------
# 5. Учёт подсказок и роли KC (primary / secondary)
# ---------------------------------------------------------------------------

def blend_with_prior(
    p_prior: float,
    p_updated: float,
    update_weight: float,   # 1.0 — полный вес, 0.4 — с подсказками
) -> float:
    """
    Смешиваем обновлённое значение с предыдущим через вес.
    update_weight=0.4: результат смещён ближе к prior → подсказка снижает кредит.
    """
    return update_weight * p_updated + (1 - update_weight) * p_prior


# ---------------------------------------------------------------------------
# Главная функция — полный шаг обновления
# ---------------------------------------------------------------------------

def update_mastery(
    *,
    current_probability: float,
    last_practiced: datetime,
    score: float,           # 0.0–1.0
    hints_used: int,
    p_transit: float,
    p_slip: float,
    p_guess: float,
    kc_role: str = "primary",       # "primary" | "secondary"
    half_life_days: float = HALF_LIFE_DAYS,
    review_mode: bool = False,      # True — не применять decay (активное освоение)
    now: datetime | None = None,
    consecutive_errors: int = 0,    # количество ошибок подряд ДО этой попытки
    consecutive_correct: int = 0,   # количество правильных ответов подряд ДО этой попытки
    recent_accuracy: float = 0.0,   # точность за последние 5 задач по этой KC (0..1)
    irt_difficulty: float | None = None,  # сложность задания [0,1] — для surprise-бонуса
    lr: float = SMOOTH_LR,          # индивидуальный estimated_lr студента
) -> float:
    """
    Полный шаг обновления: decay → smooth_update → blend.

    Использует smooth_update (EMA) вместо классического Байесовского BKT posterior.
    Это даёт монотонно убывающую скорость обучения: чем выше mastery, тем меньше
    абсолютный прирост за одну задачу.
      score=1.0, p=0.0 → +0.17   score=1.0, p=0.6 → +0.07   score=1.0, p=0.9 → +0.017

    При review_mode=True decay пропускается.
    p_transit/p_slip/p_guess принимаются для обратной совместимости, не используются.
    """
    if now is None:
        now = datetime.utcnow()

    # 1. Decay
    if review_mode:
        p_decayed = current_probability
    else:
        p_decayed = apply_decay(current_probability, last_practiced, now, half_life_days)

    # 2. Плавное обновление (EMA): Δ = effective_lr*(score − p)
    p_updated = smooth_update(
        p_decayed, score,
        lr=lr,
        consecutive_correct=consecutive_correct,
        recent_accuracy=recent_accuracy,
        irt_difficulty=irt_difficulty,
    )

    # 3. Вес обновления (hints / роль KC)
    hint_weight = 0.4 if hints_used > 0 else 1.0
    role_weight = 1.0 if kc_role == "primary" else 0.3
    update_weight = hint_weight * role_weight

    result = blend_with_prior(p_decayed, p_updated, update_weight)

    # Performance decay: систематические ошибки → активно снижаем mastery
    if consecutive_errors >= PERFORMANCE_DECAY_THRESHOLD and score < 0.5:
        result *= PERFORMANCE_DECAY_FACTOR

    return result
