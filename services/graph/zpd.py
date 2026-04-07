"""
ZPD (Zone of Proximal Development) — логика отбора KC-кандидатов.

Чистые функции без зависимостей от БД.

KC попадает в ZPD если:
  1. grade_introduced в допустимом диапазоне для ученика
  2. Все пререквизиты с strength >= STRONG_PREREQ_THRESHOLD освоены
  3. Сама KC ещё не освоена (mastery_effective < MASTERY_CEILING)
"""

from __future__ import annotations
from dataclasses import dataclass

MASTERY_THRESHOLD = 0.7     # порог освоенности пререквизита
MASTERY_CEILING = 0.95      # KC считается освоенной — убираем из ZPD
STRONG_PREREQ = 0.5         # рёбра слабее этого значения не блокируют KC


@dataclass
class Prerequisite:
    kc_id: str
    strength: float     # 0.0–1.0


@dataclass
class KCNode:
    kc_id: str
    grade_introduced: int
    difficulty_base: float
    half_life_days: float = 30.0
    subject: str = ""   # arithmetic | algebra | geometry | statistics


@dataclass
class ZPDCandidate:
    kc_id: str
    grade_introduced: int
    difficulty_base: float
    mastery_effective: float
    unmet_prereqs: list[str]
    subject: str = ""


# ---------------------------------------------------------------------------
# Grade filter
# ---------------------------------------------------------------------------

def is_grade_eligible(kc_grade: int, student_grade: int) -> bool:
    """
    Жёсткий пол:     kc_grade < student_grade - 2  →  слишком базово, пропустить
    Основная зона:   kc_grade <= student_grade      →  да
    Зона опережения: kc_grade == student_grade + 1  →  да (продвинутый контент)
    Выше:            kc_grade > student_grade + 1   →  нет
    """
    if kc_grade < student_grade - 2:
        return False
    if kc_grade > student_grade + 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def compute_zpd_candidates(
    kcs: list[KCNode],
    prerequisites: dict[str, list[Prerequisite]],   # kc_id → список пререквизитов
    mastery: dict[str, float],                       # kc_id → mastery_effective
    student_grade: int,
) -> list[ZPDCandidate]:
    """
    Возвращает KC в зоне ближайшего развития, отсортированные по приоритету:
      1. Все пререквизиты освоены (unmet_prereqs пуст)
      2. Внутри группы — по difficulty_base (от простого к сложному)

    Правило assumed mastery:
      Пререквизиты с grade_introduced < grade_floor считаются освоенными
      (ученик должен знать базовые темы предыдущих классов).
    """
    grade_floor = student_grade - 2
    kc_grade_map = {kc.kc_id: kc.grade_introduced for kc in kcs}

    candidates: list[ZPDCandidate] = []

    for kc in kcs:
        # 1. Grade filter
        if not is_grade_eligible(kc.grade_introduced, student_grade):
            continue

        # 2. KC не должна быть уже освоена
        kc_mastery = mastery.get(kc.kc_id, 0.0)
        if kc_mastery >= MASTERY_CEILING:
            continue

        # 3. Проверяем пререквизиты
        prereqs = prerequisites.get(kc.kc_id, [])
        unmet = []
        for p in prereqs:
            if p.strength < STRONG_PREREQ:
                continue
            prereq_grade = kc_grade_map.get(p.kc_id, 999)
            # Пререквизит ниже grade floor — считаем освоенным
            if prereq_grade < grade_floor:
                continue
            if mastery.get(p.kc_id, 0.0) < MASTERY_THRESHOLD:
                unmet.append(p.kc_id)

        candidates.append(ZPDCandidate(
            kc_id=kc.kc_id,
            grade_introduced=kc.grade_introduced,
            difficulty_base=kc.difficulty_base,
            mastery_effective=kc_mastery,
            unmet_prereqs=unmet,
            subject=kc.subject,
        ))

    # Сортировка: сначала ready (unmet пуст), потом по difficulty
    candidates.sort(key=lambda c: (len(c.unmet_prereqs) > 0, c.difficulty_base))
    return candidates
