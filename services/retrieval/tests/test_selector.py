"""Юнит-тесты логики выбора KC и задания. Без HTTP, без БД."""

import random
import pytest
from services.retrieval.selector import (
    ZPDEntry,
    select_kc_from_zpd,
    select_task,
    EXPLORATION_RATE,
)


def make_entry(kc_id: str, subject: str = "algebra", difficulty: float = 0.5,
               mastery: float = 0.0, ready: bool = True) -> ZPDEntry:
    return ZPDEntry(kc_id=kc_id, subject=subject, difficulty_base=difficulty,
                    mastery_effective=mastery, ready=ready)


def make_task(task_id: str, difficulty: float) -> dict:
    return {
        "task_id": task_id,
        "parts": [{"irt_difficulty": difficulty}],
    }


# ---------------------------------------------------------------------------
# select_kc_from_zpd
# ---------------------------------------------------------------------------

class TestSelectKC:
    def test_empty_candidates_returns_none(self):
        assert select_kc_from_zpd([]) is None

    def test_single_candidate_returned(self):
        entry = make_entry("kc_a")
        assert select_kc_from_zpd([entry]) == entry

    def test_ready_preferred_over_not_ready(self):
        not_ready = make_entry("kc_hard", ready=False)
        ready = make_entry("kc_easy", ready=True)
        result = select_kc_from_zpd([not_ready, ready])
        assert result == ready

    def test_subject_rotation_applied(self):
        algebra = make_entry("kc_algebra", subject="algebra")
        geometry = make_entry("kc_geometry", subject="geometry")
        # Оба готовы, последнее задание было по алгебре → выбираем геометрию
        result = select_kc_from_zpd([algebra, geometry], last_subject="algebra")
        assert result == geometry

    def test_no_rotation_if_only_same_subject(self):
        a1 = make_entry("kc_a1", subject="algebra")
        a2 = make_entry("kc_a2", subject="algebra")
        # Нет альтернативы → возвращаем один из доступных
        result = select_kc_from_zpd([a1, a2], last_subject="algebra")
        assert result in [a1, a2]

    def test_no_rotation_without_last_subject(self):
        algebra = make_entry("kc_algebra", subject="algebra")
        geometry = make_entry("kc_geometry", subject="geometry")
        result = select_kc_from_zpd([algebra, geometry], last_subject=None)
        assert result in [algebra, geometry]

    def test_ready_with_rotation(self):
        # Готовая алгебра + неготовая геометрия + готовая арифметика
        ready_alg = make_entry("kc_alg", subject="algebra", ready=True)
        not_ready_geo = make_entry("kc_geo", subject="geometry", ready=False)
        ready_arith = make_entry("kc_arith", subject="arithmetic", ready=True)
        # Последнее — алгебра. Из готовых [alg, arith] выбираем arith
        result = select_kc_from_zpd([ready_alg, not_ready_geo, ready_arith], last_subject="algebra")
        assert result == ready_arith


# ---------------------------------------------------------------------------
# select_task
# ---------------------------------------------------------------------------

class TestSelectTask:
    def test_single_task_always_returned(self):
        tasks = [make_task("t1", 0.5)]
        task, source = select_task(tasks, target_difficulty=0.5)
        assert task["task_id"] == "t1"
        assert source == "zpd"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_task([], target_difficulty=0.5)

    def test_exploitation_picks_closest_difficulty(self):
        tasks = [
            make_task("easy", 0.2),
            make_task("mid",  0.5),
            make_task("hard", 0.8),
        ]
        # target=0.45 → mid (0.5) ближайший
        rng = random.Random(42)
        # Форсируем exploitation: rng.random() должен вернуть >= EXPLORATION_RATE
        # Делаем exploration_rate=0 чтобы всегда exploitation
        task, source = select_task(tasks, target_difficulty=0.45, exploration_rate=0.0, rng=rng)
        assert task["task_id"] == "mid"
        assert source == "zpd"

    def test_exploration_without_stretch_returns_random(self):
        tasks = [make_task(f"t{i}", i * 0.1) for i in range(10)]
        # exploration_rate=1.0, stretch_difficulty=None → случайное задание
        rng = random.Random(7)
        task, source = select_task(tasks, target_difficulty=0.5, exploration_rate=1.0, rng=rng)
        assert source == "exploration"

    def test_stretch_exploration_picks_closest_to_stretch(self):
        tasks = [make_task("easy", 0.2), make_task("mid", 0.5), make_task("hard", 0.9)]
        # exploration_rate=1.0, stretch=0.85 → hard (0.9) ближайшая
        rng = random.Random(0)
        task, source = select_task(
            tasks, target_difficulty=0.3, stretch_difficulty=0.85,
            exploration_rate=1.0, rng=rng,
        )
        assert task["task_id"] == "hard"
        assert source == "stretch"

    def test_exploitation_source_label(self):
        tasks = [make_task("t1", 0.5), make_task("t2", 0.8)]
        _, source = select_task(tasks, target_difficulty=0.5, exploration_rate=0.0,
                                rng=random.Random())
        assert source == "zpd"

    def test_closest_difficulty_at_high_mastery(self):
        # mastery=0.8 → target = 0.3 + 0.4*0.8 = 0.62 → ближайшее mid (0.5)
        tasks = [make_task("easy", 0.1), make_task("mid", 0.5), make_task("hard", 0.9)]
        task, _ = select_task(tasks, target_difficulty=0.62, exploration_rate=0.0,
                               rng=random.Random())
        assert task["task_id"] == "mid"
