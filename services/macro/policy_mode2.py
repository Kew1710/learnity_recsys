"""Coverage Policy — скоринговая функция для Режима 2.

Заменяет CoverageQAgent (tabular Q-learning), который не масштабировался
на 10+ KC: пространство состояний 4^N × 5 не покрывалось за разумное число эпизодов.

Скоринг учитывает:
  gap до порога освоения  — основной сигнал
  готовность пре-реквизитов — KC недоступна если prereqs слабые
  класс ученика           — KC выше класса не предлагаются
  вариант coverage        — count / mass / frontier меняют приоритеты
  остаток бюджета         — мало задач осталось → фокус на ближайших к порогу
"""

from __future__ import annotations

MASTERY_THRESHOLD = 0.75   # порог "освоено"
PREREQ_READY_MIN = 0.50    # минимальный avg mastery prereqs чтобы KC была активна


def score_kc(
    kc_id: str,
    mastery: dict[str, float],
    prereq_masteries: list[float],
    grade_introduced: int,
    student_grade: int,
    variant: str = "count",
    budget_ratio: float = 1.0,
) -> float:
    """
    Скоринг одной KC. Выше = приоритетнее.
    Возвращает -1.0 если KC недоступна.

    Args:
        kc_id: идентификатор KC
        mastery: текущий mastery студента {kc_id: float}
        prereq_masteries: mastery прямых пре-реквизитов KC
        grade_introduced: класс в котором KC введена
        student_grade: класс студента
        variant: "count" | "mass" | "frontier"
        budget_ratio: tasks_remaining / task_budget  [0..1]
    """
    if grade_introduced > student_grade:
        return -1.0

    m = mastery.get(kc_id, 0.0)
    if m >= 0.95:
        return -1.0

    # Готовность пре-реквизитов: если prereqs слабые — KC получает низкий вес,
    # но не исключается полностью (может быть полезна как remedial)
    if prereq_masteries:
        prereq_ready = sum(prereq_masteries) / len(prereq_masteries)
        prereq_weight = min(1.0, prereq_ready / PREREQ_READY_MIN)
    else:
        prereq_weight = 1.0

    gap = max(0.0, MASTERY_THRESHOLD - m)

    if variant == "count":
        # Максимизируем число KC перешедших порог → берём самые близкие к нему
        # "дешёвые победы": mastery=0.70 нужно 1-2 задачи, mastery=0.10 — намного больше
        # max(0.05, m) чтобы новые KC (m=0) не исключались полностью, но шли последними
        base = max(0.05, m)
    elif variant == "mass":
        # Любой прирост mastery ценен, независимо от дистанции до порога
        base = 1.0 - m
    elif variant == "frontier":
        # Бонус за KC которые ещё не трогали — исследование нового
        frontier = 2.0 if m < 0.05 else 1.0
        base = (1.0 - m) * frontier
    else:
        base = max(0.05, m)

    score = base * prereq_weight

    # Мало бюджета → фокус только на ближайших к порогу, игнорируем остальное
    if budget_ratio < 0.25:
        score = max(0.05, m)

    return max(0.0, score)


def rank_coverage_kcs(
    kcs: list[dict],
    mastery: dict[str, float],
    student_grade: int,
    variant: str = "count",
    budget_ratio: float = 1.0,
    top_n: int = 10,
) -> list[str]:
    """
    Ранжирует KC по скорингу, возвращает топ-N kc_id.

    Args:
        kcs: список KC вида {kc_id, grade_introduced, prereq_masteries: [float]}
        mastery: текущий mastery студента
        student_grade: класс студента
        variant: "count" | "mass" | "frontier"
        budget_ratio: остаток бюджета [0..1]
        top_n: сколько KC включить в план
    """
    scored = [
        (kc["kc_id"], score_kc(
            kc["kc_id"],
            mastery,
            kc.get("prereq_masteries", []),
            kc["grade_introduced"],
            student_grade,
            variant,
            budget_ratio,
        ))
        for kc in kcs
    ]
    scored = [(kc_id, s) for kc_id, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [kc_id for kc_id, _ in scored[:top_n]]
