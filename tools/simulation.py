"""
Simulation engine — автоматическое тестирование рекомендательной системы.

Два вектора состояния:
  true_mastery    — реальные знания студента (не видны системе)
  visible_mastery — оценка BKT (хранится в БД, обновляется через API)

Ответ генерируется из true_mastery через IRT:
  P(correct) = p_guess + (1 - p_guess - p_slip) * sigmoid(logit(true_m) - difficulty)

true_mastery обновляется той же EMA-формулой что и в bkt.py (smooth_update).
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Константы ───────────────────────────────────────────────────────────────

MASTERY_THRESHOLD = 0.80
SMOOTH_LR      = 0.15   # EMA learning rate (зеркало bkt.py, используется visible mastery)

console = Console()


# ─── Профиль студента ────────────────────────────────────────────────────────

@dataclass
class StudentProfile:
    name: str
    initial_true_mastery: float = 0.05     # стартовое true mastery для всех KC
    growth_scale: float = 0.05             # скорость роста true mastery за правильный ответ (скрыта от системы)
    growth_scale_range: tuple = (0.03, 0.07)  # диапазон для рандомизации при создании варианта
    specific_kcs: dict[str, float] = field(default_factory=dict)  # KC-специфичные overrides
    p_slip: float = 0.10                    # знает, но ошибается (небрежность, стресс)
    p_guess: float = 0.05                   # минимальный шанс угадать (одинаков для всех)
    careless_error_rate: float = 0.05       # независимая случайная ошибка
    grade: int = 8
    description: str = ""
    # Насколько каждый тип задания субъективно сложнее (+difficulty offset).
    # procedural — базовый (0.0), остальные относительно него.
    type_difficulty_mod: dict[str, float] = field(default_factory=lambda: {
        "procedural":   0.00,
        "conceptual":   0.15,
        "word_problem": 0.10,
        "mixed":        0.20,
    })
    # Множитель на mastery gain при верном ответе по типу задания.
    # Концептуальные задания дают больше понимания если справился.
    type_learning_mod: dict[str, float] = field(default_factory=lambda: {
        "procedural":   1.0,
        "conceptual":   1.4,
        "word_problem": 1.2,
        "mixed":        1.1,
    })


PRESET_PROFILES: dict[str, StudentProfile] = {
    "fast_learner": StudentProfile(
        name="fast_learner",
        initial_true_mastery=0.05,
        growth_scale=0.10,
        growth_scale_range=(0.07, 0.14),
        p_slip=0.06,
        p_guess=0.05,
        careless_error_rate=0.03,
        grade=8,
        description="Быстро усваивает материал, мало ошибается",
        type_difficulty_mod={"procedural": 0.00, "conceptual": 0.05, "word_problem": 0.10, "mixed": 0.12},
        type_learning_mod={"procedural": 1.0, "conceptual": 1.5, "word_problem": 1.3, "mixed": 1.2},
    ),
    "advanced": StudentProfile(
        name="advanced",
        initial_true_mastery=0.70,
        growth_scale=0.07,
        growth_scale_range=(0.05, 0.10),
        p_slip=0.05,
        p_guess=0.05,
        careless_error_rate=0.02,
        grade=8,
        description="Уже знает материал (initial=0.70), система недооценивает → должна быстро откалиброваться",
        type_difficulty_mod={"procedural": 0.00, "conceptual": -0.05, "word_problem": 0.05, "mixed": 0.08},
        type_learning_mod={"procedural": 1.0, "conceptual": 1.3, "word_problem": 1.2, "mixed": 1.1},
    ),
    "average": StudentProfile(
        name="average",
        initial_true_mastery=0.08,
        growth_scale=0.06,
        growth_scale_range=(0.04, 0.09),
        p_slip=0.12,
        p_guess=0.05,
        careless_error_rate=0.05,
        grade=8,
        description="Средний темп обучения, типичный студент",
        type_difficulty_mod={"procedural": 0.00, "conceptual": 0.15, "word_problem": 0.15, "mixed": 0.20},
        type_learning_mod={"procedural": 1.0, "conceptual": 1.4, "word_problem": 1.2, "mixed": 1.1},
    ),
    "slow_learner": StudentProfile(
        name="slow_learner",
        initial_true_mastery=0.05,
        growth_scale=0.035,
        growth_scale_range=(0.020, 0.050),
        p_slip=0.22,
        p_guess=0.05,
        careless_error_rate=0.10,
        grade=8,
        description="Медленно усваивает, часто ошибается",
        type_difficulty_mod={"procedural": 0.00, "conceptual": 0.30, "word_problem": 0.25, "mixed": 0.35},
        type_learning_mod={"procedural": 1.0, "conceptual": 1.6, "word_problem": 1.3, "mixed": 1.2},
    ),
}


# ─── Шаг и результат симуляции ───────────────────────────────────────────────

@dataclass
class SimStep:
    task_num: int
    kc_id: str
    irt_difficulty: float
    score: float
    true_mastery_after: float
    visible_mastery_after: float
    recommendation_source: str
    task_type: str = "procedural"


@dataclass
class SimResult:
    profile: StudentProfile
    student_id: str
    steps: list[SimStep] = field(default_factory=list)
    final_true_mastery: dict[str, float] = field(default_factory=dict)
    final_visible_mastery: dict[str, float] = field(default_factory=dict)
    initial_true_mastery: dict[str, float] = field(default_factory=dict)
    initial_visible_mastery: dict[str, float] = field(default_factory=dict)
    time_to_mastery: dict[str, int] = field(default_factory=dict)  # kc_id → task_num when first mastered (visible)
    # План
    plan_kc_ids: list[str] = field(default_factory=list)          # KC из плана в порядке приоритета
    plan_steps_completed: int = 0                                  # шагов завершено за сессию
    plan_tasks_on_plan: int = 0                                    # заданий потрачено на KC из плана
    plan_completion_task: Optional[int] = None                     # задание на котором план полностью выполнен

    @property
    def tasks_done(self) -> int:
        return len(self.steps)

    @property
    def accuracy(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.score >= 0.9) / len(self.steps)

    @property
    def accuracy_first_half(self) -> float:
        """Accuracy первой половины сессии (исследование)."""
        half = self.steps[:len(self.steps) // 2]
        return sum(1 for s in half if s.score >= 0.9) / len(half) if half else 0.0

    @property
    def accuracy_second_half(self) -> float:
        """Accuracy второй половины сессии (эксплуатация)."""
        half = self.steps[len(self.steps) // 2:]
        return sum(1 for s in half if s.score >= 0.9) / len(half) if half else 0.0

    @property
    def practiced_kc_ids(self) -> set[str]:
        """KC по которым реально выдавались задания в этой сессии."""
        return {s.kc_id for s in self.steps}

    @property
    def mae(self) -> float:
        """MAE только по KC которые реально практиковались (задания выдавались).
        Показывает качество калибровки там где система собрала данные."""
        keys = self.practiced_kc_ids
        if not keys:
            return 1.0
        errors = [
            abs(self.final_true_mastery.get(k, 0.0) - self.final_visible_mastery.get(k, 0.0))
            for k in keys
        ]
        return sum(errors) / len(errors)

    @property
    def mae_all(self) -> float:
        """MAE по всем cold-started KC (включая непрактиковавшиеся).
        Для непрактиковавшихся visible = cold_start (0.50), true = initial_true_mastery (0.05)
        → большая ошибка показывает насколько плохо система знает студента в целом."""
        keys = set(self.final_true_mastery) | set(self.final_visible_mastery)
        if not keys:
            return 1.0
        errors = [
            abs(self.final_true_mastery.get(k, 0.0) - self.final_visible_mastery.get(k, 0.0))
            for k in keys
        ]
        return sum(errors) / len(errors)

    @property
    def mastered_true(self) -> int:
        """KC освоенные за сессию: стартовали ниже порога, финишировали выше."""
        return sum(
            1 for k, m in self.final_true_mastery.items()
            if m >= MASTERY_THRESHOLD and self.initial_true_mastery.get(k, 0.0) < MASTERY_THRESHOLD
        )

    @property
    def mastered_visible(self) -> int:
        """KC освоенные за сессию (по мнению системы): стартовали ниже порога, финишировали выше."""
        return sum(
            1 for k, m in self.final_visible_mastery.items()
            if m >= MASTERY_THRESHOLD and self.initial_visible_mastery.get(k, 0.0) < MASTERY_THRESHOLD
        )

    @property
    def plan_mastered_true(self) -> int:
        """KC из плана реально освоенные (true_mastery >= порога)."""
        return sum(
            1 for k in self.plan_kc_ids
            if self.final_true_mastery.get(k, 0.0) >= MASTERY_THRESHOLD
            and self.initial_true_mastery.get(k, 0.0) < MASTERY_THRESHOLD
        )

    @property
    def plan_mastered_visible(self) -> int:
        """KC из плана освоенные по мнению системы (visible_mastery >= порога)."""
        return sum(
            1 for k in self.plan_kc_ids
            if self.final_visible_mastery.get(k, 0.0) >= MASTERY_THRESHOLD
            and self.initial_visible_mastery.get(k, 0.0) < MASTERY_THRESHOLD
        )

    @property
    def plan_completion_rate(self) -> float:
        """Доля шагов плана завершённых за сессию."""
        if not self.plan_kc_ids:
            return 0.0
        return self.plan_steps_completed / len(self.plan_kc_ids)

    @property
    def plan_focus_rate(self) -> float:
        """Доля заданий потраченных на активный шаг плана."""
        if not self.tasks_done or not self.plan_kc_ids:
            return 0.0
        return self.plan_tasks_on_plan / self.tasks_done

    @property
    def avg_time_to_mastery(self) -> Optional[float]:
        """Среднее число заданий до первого освоения KC (visible >= 0.75)."""
        if not self.time_to_mastery:
            return None
        return sum(self.time_to_mastery.values()) / len(self.time_to_mastery)

    @property
    def unique_kcs_practiced(self) -> int:
        """Число уникальных KC, которые студент вообще практиковал."""
        return len({s.kc_id for s in self.steps})


# ─── Ядро симуляции ──────────────────────────────────────────────────────────

def simulate_answer(
    true_mastery: float,
    irt_difficulty: float,
    p_slip: float,
    p_guess: float,
    careless_error_rate: float,
    rng: random.Random,
) -> float:
    """
    3-PL IRT ответ студента.
    Возвращает 1.0, 0.5, или 0.0.

    P(correct) = p_guess + (1 - p_guess - p_slip) * sigmoid(logit(true_m) - difficulty)
    """
    m = max(0.001, min(0.999, true_mastery))
    d = max(0.001, min(0.999, irt_difficulty))
    theta = math.log(m / (1.0 - m))   # logit(mastery)
    delta = math.log(d / (1.0 - d))   # logit(difficulty) — теперь тот же масштаб
    p_irt = 1.0 / (1.0 + math.exp(-(theta - delta)))
    p_correct = p_guess + (1.0 - p_guess - p_slip) * p_irt
    p_correct = max(0.0, p_correct - careless_error_rate)

    r = rng.random()
    if r < p_correct:
        return 1.0
    # Частичный ответ: 15% от «неправильного» интервала
    elif r < p_correct + 0.15 * (1.0 - p_correct):
        return 0.5
    return 0.0


def _smooth_update(p: float, score: float, lr: float = SMOOTH_LR) -> float:
    """EMA update для visible_mastery (BKT). lr — SMOOTH_LR из bkt.py."""
    return max(0.0, min(1.0, p + lr * (score - p)))


def _true_mastery_update(p: float, score: float, growth_scale: float) -> float:
    """
    Логистический рост true_mastery — реальных знаний студента.
    Знания растут монотонно от верных ответов, не падают от неверных.

    delta = growth_scale × score × (1 − p)
      - score=1.0: максимальный прирост, замедляется по мере роста p
      - score=0.0: знания не меняются
      - score=0.5: частичный прирост
    """
    delta = growth_scale * score * (1.0 - p)
    return min(1.0, p + delta)


def run_single(
    client: httpx.Client,
    profile: StudentProfile,
    max_tasks: int = 100,
    rng: Optional[random.Random] = None,
    on_step: Optional[Callable[[SimStep, dict[str, float], dict[str, float]], None]] = None,
    session_tag: str = "",
    plan_config: Optional[dict] = None,  # {"mode": "target_mastery"|"coverage", ...параметры macro}
) -> SimResult:
    """
    Прогоняет симуляцию для одного студента с заданным профилем.

    Args:
        client:    httpx.Client с запущенными сервисами
        profile:   StudentProfile с параметрами студента
        max_tasks: максимальное число заданий
        rng:       random.Random (или None → new instance)
        on_step:   колбэк (step, true_mastery, visible_mastery) для live-отображения
    """
    if rng is None:
        rng = random.Random()

    from tools.play import (
        GATEWAY_URL, PROFILE_URL, TIMEOUT,
        api_create_student,
        api_first_task as api_next_task,
        api_full_mastery as api_mastery,
        api_submit as _api_submit_raw,
    )

    def api_submit(cl, sid, rec, score):
        part = rec["task"]["parts"][0]
        return _api_submit_raw(
            cl, sid,
            task_id=str(rec["task"]["task_id"]),
            part_id=str(part["part_id"]),
            score=score,
            primary_kcs=rec.get("primary_kcs", [rec["kc_id"]]),
            half_life_days=rec.get("half_life_days", 30.0),
            recommendation_source=rec.get("recommendation_source", "thompson_sampling"),
            last_subject=rec.get("subject"),
        )

    # 1. Создаём нового студента (gateway автоматически делает cold_start)
    student_data = api_create_student(client, profile.grade)
    student_id = student_data["student_id"]

    # 1b. Генерируем план через macro сервис если передан plan_config
    plan_kc_set: set[str] = set()
    plan_id: str | None = None
    plan_steps: list[str] = []   # KC в порядке выполнения (priority DESC)
    plan_step_idx: int = 0       # индекс текущего активного шага
    if plan_config:
        try:
            from tools.run_ab import MACRO_URL
            payload = {"student_id": student_id, **plan_config}
            r = client.post(f"{MACRO_URL}/plans", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            plan_resp = r.json()
            plan_id = plan_resp.get("plan_id")
            # Читаем шаги плана чтобы знать какие KC отслеживать
            if plan_id:
                r2 = client.get(f"{MACRO_URL}/plans/{plan_id}", timeout=TIMEOUT)
                if r2.status_code == 200:
                    plan_data = r2.json()
                    plan_steps = [s["kc_id"] for s in plan_data.get("steps", [])]
                    plan_kc_set = set(plan_steps)
        except Exception:
            pass  # план не критичен — продолжаем без него

    # 2. Инициализируем true mastery из cold_start значений.
    # После создания студента gateway уже проставил visible mastery на основе класса:
    #   KC grade+2 → 0.8,  KC grade+1 → 0.65,  остальные → 0.
    # visible_mastery = cold_start prior системы (что система думает о студенте).
    # true_mastery    = реальный студент: profile.initial_true_mastery для всех KC,
    #                   profile.specific_kcs перекрывает конкретные KC.
    # Задача системы — через задания обнаружить расхождение и подтянуть visible к true.
    # visible_mastery = cold_start prior системы (что система думает о студенте).
    # true_mastery:
    #   - KC прошлых классов (cold_start > 0.50): студент знает их так же как ожидает система
    #   - KC текущего класса (cold_start == 0.50): реальная mastery = initial_true_mastery
    #     (вот здесь система должна открывать студента через задания)
    # profile.specific_kcs перекрывает всё.
    visible_mastery: dict[str, float] = {}
    try:
        for rec in api_mastery(client, student_id):
            visible_mastery[rec["kc_id"]] = rec["probability"]
    except Exception:
        pass

    true_mastery: dict[str, float] = {
        kc_id: (profile.initial_true_mastery if vis <= 0.50 else vis)
        for kc_id, vis in visible_mastery.items()
    }
    true_mastery.update(profile.specific_kcs)
    result = SimResult(profile=profile, student_id=student_id)
    if plan_kc_set:
        result.plan_kc_ids = list(plan_kc_set)
    result.initial_true_mastery = dict(true_mastery)
    result.initial_visible_mastery = dict(visible_mastery)

    last_subject: Optional[str] = None

    for task_num in range(1, max_tasks + 1):
        # Получаем рекомендацию
        try:
            rec = api_next_task(client, student_id, last_subject)
        except Exception:
            break

        kc_id = rec["kc_id"]
        part = rec["task"]["parts"][0]
        irt_diff = part.get("irt_difficulty", 0.5)
        task_type = part.get("task_type", "procedural")
        last_subject = rec.get("subject")

        # Симулируем ответ из TRUE mastery с поправкой на тип задания
        true_m = true_mastery.get(kc_id, profile.initial_true_mastery)
        type_diff_offset = profile.type_difficulty_mod.get(task_type, 0.0)
        effective_diff = min(0.99, max(0.01, irt_diff + type_diff_offset))
        score = simulate_answer(true_m, effective_diff, profile.p_slip, profile.p_guess,
                                profile.careless_error_rate, rng)

        # Отправляем ответ в систему (обновляет visible mastery в БД)
        try:
            resp = api_submit(client, student_id, rec, score)
        except Exception:
            break

        # Извлекаем visible mastery из ответа
        mastery_update = resp.get("mastery_update", {})
        updated = mastery_update.get("updated_mastery", {})
        for kc, v in updated.items():
            visible_mastery[kc] = v
            if v >= MASTERY_THRESHOLD and kc not in result.time_to_mastery:
                result.time_to_mastery[kc] = task_num

        # Трекаем задания на активный шаг плана (focus)
        if plan_steps and plan_step_idx < len(plan_steps):
            if kc_id == plan_steps[plan_step_idx]:
                result.plan_tasks_on_plan += 1

        # Продвигаем шаг плана если текущий KC освоен (замена Kafka-consumer в симуляции)
        if plan_id and plan_steps and plan_step_idx < len(plan_steps):
            current_step_kc = plan_steps[plan_step_idx]
            if visible_mastery.get(current_step_kc, 0.0) >= MASTERY_THRESHOLD:
                try:
                    from tools.run_ab import MACRO_URL
                    client.post(f"{MACRO_URL}/plans/{plan_id}/advance", timeout=TIMEOUT)
                except Exception:
                    pass
                plan_step_idx += 1
                result.plan_steps_completed = plan_step_idx
                if plan_step_idx == len(plan_steps) and result.plan_completion_task is None:
                    result.plan_completion_task = task_num

        # Обновляем TRUE mastery — логистический рост от верных ответов.
        # Conceptual/word_problem дают больший прирост знаний (type_learning_mod).
        type_lr_mult = profile.type_learning_mod.get(task_type, 1.0)
        effective_gs = profile.growth_scale * type_lr_mult
        new_true = _true_mastery_update(true_m, score, effective_gs)
        true_mastery[kc_id] = new_true

        step = SimStep(
            task_num=task_num,
            kc_id=kc_id,
            irt_difficulty=irt_diff,
            score=score,
            true_mastery_after=new_true,
            visible_mastery_after=visible_mastery.get(kc_id, 0.0),
            recommendation_source=rec.get("recommendation_source", ""),
            task_type=task_type,
        )
        result.steps.append(step)

        if on_step:
            on_step(step, dict(true_mastery), dict(visible_mastery))

    result.final_true_mastery = dict(true_mastery)
    result.final_visible_mastery = dict(visible_mastery)
    _save_result_log(result, session_tag=session_tag)
    return result


SIM_LOG_DIR = Path(__file__).parent.parent / "logs" / "sim"


def _save_result_log(result: SimResult, session_tag: str = "") -> None:
    """
    Сохраняет полную траекторию симуляции в JSONL.
    Файл: logs/sim/<session_tag>_<profile>.jsonl  (или YYYYMMDD_HHMMSS_<profile>.jsonl)
    """
    SIM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = session_tag if session_tag else ts
    path = SIM_LOG_DIR / f"{prefix}_{result.profile.name}.jsonl"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_tag": session_tag or ts,
        "student_id": result.student_id,
        "profile": {
            "name": result.profile.name,
            "initial_true_mastery": result.profile.initial_true_mastery,
            "growth_scale": result.profile.growth_scale,
            "p_slip": result.profile.p_slip,
            "p_guess": result.profile.p_guess,
            "careless_error_rate": result.profile.careless_error_rate,
            "grade": result.profile.grade,
        },
        "summary": {
            "tasks_done": result.tasks_done,
            "accuracy": round(result.accuracy, 4),
            "accuracy_first_half": round(result.accuracy_first_half, 4),
            "accuracy_second_half": round(result.accuracy_second_half, 4),
            "mae": round(result.mae, 4),
            "mae_all": round(result.mae_all, 4),
            "mastered_true": result.mastered_true,
            "mastered_visible": result.mastered_visible,
            "avg_time_to_mastery": round(result.avg_time_to_mastery, 1) if result.avg_time_to_mastery else None,
            "unique_kcs": result.unique_kcs_practiced,
            "plan_steps_completed": result.plan_steps_completed,
            "plan_total_steps": len(result.plan_kc_ids),
            "plan_mastered_true": result.plan_mastered_true,
            "plan_mastered_visible": result.plan_mastered_visible,
            "plan_completion_rate": round(result.plan_completion_rate, 4),
            "plan_focus_rate": round(result.plan_focus_rate, 4),
            "plan_completion_task": result.plan_completion_task,
        },
        "final_true_mastery": {k: round(v, 4) for k, v in result.final_true_mastery.items()},
        "final_visible_mastery": {k: round(v, 4) for k, v in result.final_visible_mastery.items()},
        "time_to_mastery": result.time_to_mastery,
        "trajectory": [
            {
                "t": s.task_num,
                "kc": s.kc_id,
                "type": s.task_type,
                "diff": round(s.irt_difficulty, 3),
                "score": s.score,
                "true_m": round(s.true_mastery_after, 4),
                "vis_m": round(s.visible_mastery_after, 4),
                "src": s.recommendation_source,
            }
            for s in result.steps
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)


def _run_clustering_sync() -> bool:
    """
    Запускает кластеризацию в отдельном subprocess.
    Subprocess получает чистый event loop и свои DB-соединения —
    это избегает конфликта с event loop текущего процесса.
    """
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import asyncio; from services.clustering.cluster import run_clustering; asyncio.run(run_clustering())",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        console.print(f"  [yellow]⚠ кластеризация: {err[-1] if err else 'unknown error'}[/yellow]")
        return False
    return True


def run_population(
    client: httpx.Client,
    profiles: list[StudentProfile],
    max_tasks: int = 100,
    rng_seed: Optional[int] = None,
    cluster_every: int = 5,
    session_tag: str = "",
    plan_config: Optional[dict] = None,
) -> list[SimResult]:
    """
    Запускает симуляцию для каждого профиля последовательно.
    Каждые cluster_every студентов запускает кластеризацию.
    Возвращает список результатов в том же порядке что profiles.
    """
    results: list[SimResult] = []
    base_rng = random.Random(rng_seed)

    for i, profile in enumerate(profiles):
        profile_rng = random.Random(base_rng.randint(0, 2**32))

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[dim]{profile.name}[/dim] {{task.completed}}/{max_tasks}"),
            BarColumn(bar_width=25),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("", total=max_tasks)

            def _on_step(step, tm, vm, _prog=progress, _tid=task_id):
                _prog.update(_tid, completed=step.task_num)

            result = run_single(client, profile, max_tasks, profile_rng, on_step=_on_step, session_tag=session_tag, plan_config=plan_config)

        results.append(result)
        plan_str = ""
        if result.plan_kc_ids:
            plan_str = (
                f"  [cyan]план:[/cyan] {result.plan_steps_completed}/{len(result.plan_kc_ids)} шагов"
                f"  focus: {result.plan_focus_rate*100:.0f}%"
                + (f"  done@{result.plan_completion_task}" if result.plan_completion_task else "")
            )
        console.print(
            f"  [green]✓[/green] [bold]{profile.name}[/bold]  grade={profile.grade}  "
            f"{result.tasks_done} заданий\n"
            f"    acc: {result.accuracy*100:.0f}%"
            f"  (1я пол: {result.accuracy_first_half*100:.0f}%"
            f" → 2я пол: {result.accuracy_second_half*100:.0f}%)\n"
            f"    MAE(seen): {result.mae:.3f}  MAE(all): {result.mae_all:.3f}\n"
            f"    освоено true/visible: {result.mastered_true}/{result.mastered_visible}"
            + ("\n   " + plan_str if result.plan_kc_ids else "")
        )

        # Кластеризация каждые cluster_every студентов
        if (i + 1) % cluster_every == 0:
            console.print(f"  [dim cyan]↺ кластеризация после {i+1} студентов...[/dim cyan]")
            _run_clustering_sync()

    # Финальная кластеризация после всех студентов
    if len(profiles) % cluster_every != 0:
        console.print(f"  [dim cyan]↺ финальная кластеризация...[/dim cyan]")
        _run_clustering_sync()

    return results


GRADE_POOL = [5, 7, 8, 9]


def _sample_profile_ab(base: StudentProfile, i: int, rng: random.Random) -> StudentProfile:
    """
    Создаёт вариацию профиля с разбросом параметров и случайным классом из GRADE_POOL.
    Используется для A/B-эксперимента с разными классами.
    """
    p = _sample_profile(base, i, rng)
    p.grade = rng.choice(GRADE_POOL)
    return p


def run_warmup(
    client: httpx.Client,
    n_per_group: int = 20,
    max_tasks: int = 60,
) -> None:
    """
    Прогрев кластеров перед основной симуляцией.
    Прогоняет n_per_group студентов из каждой группы с max_tasks заданий.
    Цель: заполнить bandit_log и инициализировать кластеры до основного теста.
    """
    console.print()
    console.print(Panel(
        f"[bold]Прогрев кластеров[/bold]  [dim]— {n_per_group} студентов × 4 группы × {max_tasks} заданий[/dim]\n\n"
        "  Заполняет bandit_log и инициализирует кластеры перед основным тестом.\n"
        "  Студенты прогрева не входят в итоговую статистику.",
        border_style="yellow dim", padding=(0, 2),
    ))

    base_rng = random.Random(999)
    profiles: list[StudentProfile] = []
    for base in PRESET_PROFILES.values():
        profiles.extend(_sample_profile_ab(base, i, base_rng) for i in range(n_per_group))

    total = len(profiles)
    console.print(f"[dim]Запускаю {total} прогревочных студентов...[/dim]\n")
    run_population(client, profiles, max_tasks=max_tasks, cluster_every=5)
    console.print("[green]✓ Прогрев завершён. Кластеры инициализированы.[/green]\n")


# ─── Отображение результатов ─────────────────────────────────────────────────

def _mastery_bar(v: float, width: int = 8) -> str:
    filled = round(v * width)
    color = "green" if v >= MASTERY_THRESHOLD else ("yellow" if v >= 0.40 else "red")
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/{color}] {v:.2f}"


def show_single_result(result: SimResult) -> None:
    """Rich-вывод результата симуляции одного студента."""
    console.print()
    console.print(Panel(
        f"[bold]Профиль:[/bold] [cyan]{result.profile.name}[/cyan]  "
        f"[dim]{result.profile.description}[/dim]\n\n"
        f"  student_id:  {result.student_id}\n"
        f"  Заданий:     {result.tasks_done}\n"
        f"  Точность:    [bold]{result.accuracy*100:.0f}%[/bold]\n"
        f"  MAE(true↔visible): [{'green' if result.mae < 0.15 else 'yellow'}]{result.mae:.3f}[/{'green' if result.mae < 0.15 else 'yellow'}]\n"
        f"  KC освоено (true / visible): "
        f"[bold]{result.mastered_true}[/bold] / [bold]{result.mastered_visible}[/bold]",
        border_style="cyan dim", padding=(0, 2),
    ))

    # Таблица по KC: true vs visible mastery
    all_kcs = sorted(set(result.final_true_mastery) | set(result.final_visible_mastery))
    if not all_kcs:
        console.print("[dim]Нет отслеженных KC[/dim]")
        return

    t = Table(
        title=f"KC mastery: true vs visible ({len(all_kcs)} KC)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    t.add_column("KC", style="dim", max_width=32)
    t.add_column("True mastery")
    t.add_column("Visible mastery")
    t.add_column("|Δ|", justify="right")

    kcs_sorted = sorted(
        all_kcs,
        key=lambda k: result.final_true_mastery.get(k, 0.0),
        reverse=True,
    )

    for kc in kcs_sorted:
        tm = result.final_true_mastery.get(kc, 0.0)
        vm = result.final_visible_mastery.get(kc, 0.0)
        delta = abs(tm - vm)
        delta_color = "red" if delta > 0.25 else ("yellow" if delta > 0.10 else "green")
        t.add_row(kc, _mastery_bar(tm), _mastery_bar(vm), f"[{delta_color}]{delta:.3f}[/{delta_color}]")

    console.print(t)

    # Кривая обучения по задачам (каждые 10 задач)
    if len(result.steps) >= 10:
        _show_learning_curve(result)


def _show_learning_curve(result: SimResult) -> None:
    """Показывает кривую true vs visible mastery для основной KC (самой активной)."""
    # Ищем KC с наибольшим числом задач
    kc_counts: dict[str, int] = {}
    for s in result.steps:
        kc_counts[s.kc_id] = kc_counts.get(s.kc_id, 0) + 1

    top_kc = max(kc_counts, key=lambda k: kc_counts[k])

    kc_steps = [s for s in result.steps if s.kc_id == top_kc]
    if len(kc_steps) < 3:
        return

    console.print(f"\n  [dim]Кривая обучения для [bold]{top_kc}[/bold] ({len(kc_steps)} задач):[/dim]")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    t.add_column("#", width=4, justify="right", style="dim")
    t.add_column("Скор", width=5, justify="center")
    t.add_column("True mastery")
    t.add_column("Visible mastery")

    # Показываем каждые N шагов (не более 15 строк)
    step_every = max(1, len(kc_steps) // 15)
    for i, s in enumerate(kc_steps):
        if i % step_every != 0 and i != len(kc_steps) - 1:
            continue
        score_str = (
            "[green]1.0[/green]" if s.score >= 0.9
            else "[yellow]0.5[/yellow]" if s.score >= 0.4
            else "[red]0.0[/red]"
        )
        t.add_row(
            str(s.task_num),
            score_str,
            _mastery_bar(s.true_mastery_after, 10),
            _mastery_bar(s.visible_mastery_after, 10),
        )

    console.print(t)


def _sample_profile(base: StudentProfile, i: int, rng: random.Random) -> StudentProfile:
    """
    Создаёт вариацию профиля с небольшим разбросом параметров.
    Каждый студент из одной группы немного отличается.
    """
    def jitter(v: float, scale: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v + rng.gauss(0, scale)))

    lo, hi = base.growth_scale_range
    return StudentProfile(
        name=f"{base.name}_{i+1}",
        initial_true_mastery=jitter(base.initial_true_mastery, 0.05, 0.0, 0.95),
        growth_scale=rng.uniform(lo, hi),
        growth_scale_range=base.growth_scale_range,
        specific_kcs=dict(base.specific_kcs),
        p_slip=jitter(base.p_slip, 0.02, 0.01, 0.45),
        p_guess=base.p_guess,           # одинаков для всех, не джиттерим
        careless_error_rate=jitter(base.careless_error_rate, 0.01, 0.0, 0.20),
        grade=base.grade,
        description=base.description,
    )


def show_cluster_alignment(results: list[SimResult]) -> None:
    """
    Показывает таблицу соответствия кластеров и профильных групп.

    Запрашивает cluster_id каждого студента из БД и строит матрицу:
      строки  = профильные группы (fast_learner, slow_learner, ...)
      столбцы = номера кластеров
      ячейка  = сколько студентов этой группы попало в этот кластер
    """
    import asyncio
    from shared.db import AsyncSessionLocal
    import sqlalchemy as sa

    import subprocess, json as _json
    fetch_result = subprocess.run(
        [
            sys.executable, "-c",
            "import asyncio, json; from shared.db import AsyncSessionLocal; import sqlalchemy as sa\n"
            "async def f():\n"
            "    async with AsyncSessionLocal() as db:\n"
            "        rows = (await db.execute(sa.text('SELECT student_id::text, cluster_id FROM student_clusters'))).fetchall()\n"
            "    print(json.dumps({r[0]: r[1] for r in rows}))\n"
            "asyncio.run(f())",
        ],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    try:
        cluster_map = _json.loads(fetch_result.stdout.strip())
    except Exception as e:
        console.print(f"  [yellow]Не удалось загрузить кластеры: {e}[/yellow]")
        return

    if not cluster_map:
        console.print("  [yellow]Кластеры не назначены — запустите прогрев или симуляцию с достаточным числом студентов.[/yellow]")
        return

    # Группируем студентов из results по профилю
    def _base(name: str) -> str:
        parts = name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

    # Собираем какие кластеры встретились
    all_clusters = sorted({c for c in cluster_map.values()})
    groups_order = list(dict.fromkeys(_base(r.profile.name) for r in results))

    # counts[group][cluster] = count
    counts: dict[str, dict[int, int]] = {g: {} for g in groups_order}
    unassigned = 0
    for r in results:
        grp = _base(r.profile.name)
        cid = cluster_map.get(r.student_id)
        if cid is None:
            unassigned += 1
            continue
        counts[grp][cid] = counts[grp].get(cid, 0) + 1

    # Находим "доминирующий" кластер для каждой группы
    dominant: dict[str, int | None] = {}
    for g in groups_order:
        if counts[g]:
            dominant[g] = max(counts[g], key=lambda c: counts[g][c])
        else:
            dominant[g] = None

    console.print()
    t = Table(
        title="Соответствие кластеров и профильных групп",
        box=box.ROUNDED, show_header=True, header_style="bold dim",
    )
    t.add_column("Группа", style="bold", min_width=16)
    t.add_column("Доминирующий кластер", justify="center")
    for c in all_clusters:
        t.add_column(f"кл.{c}", justify="right")
    t.add_column("итого", justify="right", style="dim")

    for g in groups_order:
        dom = dominant[g]
        dom_str = f"[green]#{dom}[/green]" if dom is not None else "[dim]—[/dim]"
        total_g = sum(counts[g].values())
        cells = []
        for c in all_clusters:
            cnt = counts[g].get(c, 0)
            if cnt == 0:
                cells.append("[dim]·[/dim]")
            elif c == dom:
                cells.append(f"[green bold]{cnt}[/green bold]")
            else:
                cells.append(str(cnt))
        t.add_row(g, dom_str, *cells, str(total_g))

    console.print(t)

    # Насколько хорошо кластеры разделяют группы
    purity_scores = []
    for g in groups_order:
        total_g = sum(counts[g].values())
        if total_g > 0 and dominant[g] is not None:
            purity_scores.append(counts[g][dominant[g]] / total_g)

    if purity_scores:
        avg_purity = sum(purity_scores) / len(purity_scores)
        color = "green" if avg_purity >= 0.7 else ("yellow" if avg_purity >= 0.5 else "red")
        console.print(
            f"\n  Чистота кластеров (cluster purity): "
            f"[{color}]{avg_purity:.0%}[/{color}]  "
            f"[dim](100% = каждая группа полностью в своём кластере)[/dim]"
        )

    if unassigned:
        console.print(f"  [dim yellow]{unassigned} студентов без кластера[/dim yellow]")


def _group_stats(results: list[SimResult]) -> str:
    """Форматирует среднее ± std для группы результатов."""
    def _fmt(vals: list[float]) -> str:
        avg = sum(vals) / len(vals)
        std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        return f"{avg:.2f}±{std:.2f}"

    ttm_vals = [r.avg_time_to_mastery for r in results if r.avg_time_to_mastery is not None]
    ttm_str = f"ttm {_fmt(ttm_vals)}" if ttm_vals else "ttm —"

    return (
        f"точность {_fmt([r.accuracy for r in results])}  "
        f"MAE(вид) {_fmt([r.mae for r in results])}  "
        f"MAE(все) {_fmt([r.mae_all for r in results])}  "
        f"освоено true {_fmt([float(r.mastered_true) for r in results])}  "
        f"vis {_fmt([float(r.mastered_visible) for r in results])}  "
        f"{ttm_str}  "
        f"KC {_fmt([float(r.unique_kcs_practiced) for r in results])}"
    )


def show_population_results(
    results: list[SimResult],
    group_name: Optional[str] = None,
    grouped: bool = False,
) -> None:
    """
    Сравнительная таблица результатов.

    grouped=True: результаты содержат студентов из разных групп —
    показываем каждого плюс итоговую строку по каждой группе.
    """
    console.print()

    title = f"Группа: {group_name}" if group_name else "Результаты симуляции"
    t = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold dim")
    t.add_column("Студент", style="bold", min_width=18)
    t.add_column("Задания", justify="right")
    t.add_column("Точность", justify="right")
    t.add_column("KC", justify="right")
    t.add_column("Освоено true", justify="right")
    t.add_column("Освоено vis", justify="right")
    t.add_column("MAE (вид.)", justify="right")
    t.add_column("MAE (все)", justify="right")

    # Группируем по base-имени профиля (strip trailing _N)
    def _base(name: str) -> str:
        parts = name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

    from itertools import groupby
    prev_base = None

    for r in results:
        base = _base(r.profile.name)
        # Разделитель между группами
        if grouped and prev_base is not None and base != prev_base:
            t.add_section()
        prev_base = base

        mae_color = "green" if r.mae < 0.15 else ("yellow" if r.mae < 0.30 else "red")
        mae_all_color = "green" if r.mae_all < 0.15 else ("yellow" if r.mae_all < 0.30 else "red")
        acc_color = "green" if r.accuracy > 0.7 else ("yellow" if r.accuracy > 0.5 else "red")
        n_kcs = len(set(r.final_true_mastery) | set(r.final_visible_mastery))
        t.add_row(
            r.profile.name,
            str(r.tasks_done),
            f"[{acc_color}]{r.accuracy*100:.0f}%[/{acc_color}]",
            str(n_kcs),
            str(r.mastered_true),
            str(r.mastered_visible),
            f"[{mae_color}]{r.mae:.3f}[/{mae_color}]",
            f"[{mae_all_color}]{r.mae_all:.3f}[/{mae_all_color}]",
        )

    console.print(t)

    # Агрегированная статистика
    if len(results) > 1:
        if grouped:
            # Отдельная строка на каждую группу
            by_group: dict[str, list[SimResult]] = {}
            for r in results:
                by_group.setdefault(_base(r.profile.name), []).append(r)

            console.print("\n  [dim bold]Среднее по группам:[/dim bold]")
            for grp, grp_results in by_group.items():
                console.print(f"  [cyan]{grp}[/cyan] (n={len(grp_results)}): {_group_stats(grp_results)}")
        else:
            console.print(f"\n  [dim]Итого (n={len(results)}): {_group_stats(results)}[/dim]")

    console.print()


# ─── A/B сохранение и сравнение ─────────────────────────────────────────────

AB_SUMMARY_DIR = SIM_LOG_DIR / "ab"


def save_session_summary(
    results: list[SimResult],
    variant: str,
    session_tag: str,
    rng_seed: int,
    max_tasks: int,
) -> Path:
    """
    Сохраняет агрегированный JSON-отчёт по всему прогону A/B-эксперимента.
    Файл: logs/sim/ab/<session_tag>_summary.json
    """
    AB_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    def _base(name: str) -> str:
        parts = name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

    def _agg(vals: list[float]) -> dict:
        if not vals:
            return {}
        avg = sum(vals) / len(vals)
        std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        return {"mean": round(avg, 4), "std": round(std, 4), "n": len(vals)}

    by_group: dict[str, list[SimResult]] = {}
    for r in results:
        by_group.setdefault(_base(r.profile.name), []).append(r)

    groups_agg = {}
    for grp, grp_results in by_group.items():
        ttm_vals = [r.avg_time_to_mastery for r in grp_results if r.avg_time_to_mastery is not None]
        groups_agg[grp] = {
            "n": len(grp_results),
            "accuracy": _agg([r.accuracy for r in grp_results]),
            "accuracy_first_half": _agg([r.accuracy_first_half for r in grp_results]),
            "accuracy_second_half": _agg([r.accuracy_second_half for r in grp_results]),
            "mae": _agg([r.mae for r in grp_results]),
            "mae_all": _agg([r.mae_all for r in grp_results]),
            "mastered_true": _agg([float(r.mastered_true) for r in grp_results]),
            "mastered_visible": _agg([float(r.mastered_visible) for r in grp_results]),
            "avg_time_to_mastery": _agg(ttm_vals) if ttm_vals else {},
            "unique_kcs": _agg([float(r.unique_kcs_practiced) for r in grp_results]),
        }

    ttm_all = [r.avg_time_to_mastery for r in results if r.avg_time_to_mastery is not None]
    summary = {
        "variant": variant,
        "session_tag": session_tag,
        "ts": datetime.now(timezone.utc).isoformat(),
        "rng_seed": rng_seed,
        "max_tasks": max_tasks,
        "n_students": len(results),
        "overall": {
            "accuracy": _agg([r.accuracy for r in results]),
            "accuracy_first_half": _agg([r.accuracy_first_half for r in results]),
            "accuracy_second_half": _agg([r.accuracy_second_half for r in results]),
            "mae": _agg([r.mae for r in results]),
            "mae_all": _agg([r.mae_all for r in results]),
            "mastered_true": _agg([float(r.mastered_true) for r in results]),
            "mastered_visible": _agg([float(r.mastered_visible) for r in results]),
            "avg_time_to_mastery": _agg(ttm_all) if ttm_all else {},
            "unique_kcs": _agg([float(r.unique_kcs_practiced) for r in results]),
        },
        "by_group": groups_agg,
    }

    path = AB_SUMMARY_DIR / f"{session_tag}_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n  [dim green]✓ Сводка сохранена: {path.relative_to(Path(__file__).parent.parent)}[/dim green]")
    return path


def show_ab_comparison() -> None:
    """
    Загружает два последних A/B summary-файла (linucb и baseline) и показывает сравнение.
    Ищет в logs/sim/ab/ файлы с 'linucb' и 'baseline' в имени.
    """
    if not AB_SUMMARY_DIR.exists():
        console.print("  [yellow]Нет сохранённых A/B результатов (logs/sim/ab/)[/yellow]")
        return

    def _latest(variant: str) -> Optional[dict]:
        files = sorted(AB_SUMMARY_DIR.glob(f"*{variant}*_summary.json"), reverse=True)
        if not files:
            return None
        with files[0].open(encoding="utf-8") as f:
            data = json.load(f)
        console.print(f"  [dim]Загружен: {files[0].name}[/dim]")
        return data

    a = _latest("linucb")
    b = _latest("baseline")

    if not a and not b:
        console.print("  [yellow]Файлы A/B не найдены. Сначала запусти режимы 6 и 7.[/yellow]")
        return

    console.print()
    console.print(Panel(
        "[bold]A/B сравнение: LinUCB vs Baseline[/bold]\n\n"
        + (f"  LinUCB:   seed={a['rng_seed']}  n={a['n_students']}  tasks={a['max_tasks']}  [{a['session_tag']}]\n" if a else "  LinUCB:   [dim]нет данных[/dim]\n")
        + (f"  Baseline: seed={b['rng_seed']}  n={b['n_students']}  tasks={b['max_tasks']}  [{b['session_tag']}]" if b else "  Baseline: [dim]нет данных[/dim]"),
        border_style="cyan", padding=(0, 2),
    ))

    metrics = [
        ("accuracy",           "Точность",        True),
        ("mae",                "MAE (видимые KC)", False),
        ("mae_all",            "MAE (все KC)",     False),
        ("mastered_visible",   "Освоено KC (vis)", True),
        ("avg_time_to_mastery","Time-to-Mastery",  False),
        ("unique_kcs",         "Уникальных KC",    True),
    ]

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold dim")
    t.add_column("Метрика", style="bold", min_width=20)
    t.add_column("LinUCB", justify="right")
    t.add_column("Baseline", justify="right")
    t.add_column("Δ (A−B)", justify="right")

    def _fmt_agg(d: dict) -> str:
        if not d:
            return "[dim]—[/dim]"
        return f"{d['mean']:.3f} ±{d['std']:.3f}"

    for key, label, higher_is_better in metrics:
        av = (a or {}).get("overall", {}).get(key, {})
        bv = (b or {}).get("overall", {}).get(key, {})
        a_mean = av.get("mean") if av else None
        b_mean = bv.get("mean") if bv else None

        if a_mean is not None and b_mean is not None:
            delta = a_mean - b_mean
            better = (delta > 0) == higher_is_better
            delta_str = f"[{'green' if better else 'red'}]{delta:+.3f}[/{'green' if better else 'red'}]"
        else:
            delta_str = "[dim]—[/dim]"

        t.add_row(label, _fmt_agg(av), _fmt_agg(bv), delta_str)

    console.print(t)

    # По группам
    all_groups = sorted({g for d in [a, b] if d for g in d.get("by_group", {})})
    if all_groups:
        console.print("\n  [dim bold]Time-to-Mastery по группам:[/dim bold]")
        tg = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        tg.add_column("Группа", style="dim", min_width=18)
        tg.add_column("LinUCB ttm", justify="right")
        tg.add_column("Baseline ttm", justify="right")
        tg.add_column("Δ", justify="right")

        for grp in all_groups:
            ag = (a or {}).get("by_group", {}).get(grp, {}).get("avg_time_to_mastery", {})
            bg = (b or {}).get("by_group", {}).get(grp, {}).get("avg_time_to_mastery", {})
            am = ag.get("mean") if ag else None
            bm = bg.get("mean") if bg else None
            if am is not None and bm is not None:
                delta = am - bm
                delta_str = f"[{'green' if delta < 0 else 'red'}]{delta:+.1f}[/{'green' if delta < 0 else 'red'}]"
            else:
                delta_str = "[dim]—[/dim]"
            tg.add_row(
                grp,
                f"{am:.1f}" if am else "[dim]—[/dim]",
                f"{bm:.1f}" if bm else "[dim]—[/dim]",
                delta_str,
            )
        console.print(tg)
        console.print("  [dim](Δ<0 = LinUCB быстрее достигает mastery)[/dim]")


# ─── Wizard для sandbox ──────────────────────────────────────────────────────

def _safe_input(prompt: str) -> str:
    try:
        return console.input(prompt)
    except UnicodeDecodeError:
        import sys
        raw = sys.stdin.buffer.readline()
        return raw.decode("utf-8", errors="replace").rstrip("\n")


def _print_profiles_list() -> None:
    for i, (name, p) in enumerate(PRESET_PROFILES.items(), 1):
        console.print(f"  [bold]{i}[/bold] — [cyan]{name}[/cyan]: {p.description}")


def _pick_profile(prompt: str = "  Профиль (1–4, Enter=1): ") -> StudentProfile:
    _print_profiles_list()
    profile_raw = _safe_input(prompt).strip()
    profile_names = list(PRESET_PROFILES.keys())
    try:
        idx = int(profile_raw) - 1
        return PRESET_PROFILES[profile_names[idx]] if 0 <= idx < len(profile_names) else PRESET_PROFILES["fast_learner"]
    except (ValueError, IndexError):
        return PRESET_PROFILES["fast_learner"]


def sim_wizard(client: httpx.Client) -> None:
    """
    Интерактивный мастер запуска симуляции из sandbox.
    Вызывается командой 'sim'.
    """
    console.print()
    console.print(Panel(
        "[bold]Симуляция студента[/bold]  [dim]— автоматический прогон без ручного ввода[/dim]\n\n"
        "  Система создаёт нового студента, симулирует ответы из true mastery (IRT)\n"
        "  и обновляет visible mastery через BKT.\n\n"
        "  Метрика: MAE(true, visible) — насколько точно система знает студента.",
        border_style="magenta dim", padding=(0, 2),
    ))

    console.print("  [bold]1[/bold] — Один студент         [dim]детальный отчёт с кривой обучения[/dim]")
    console.print("  [bold]2[/bold] — Одна группа × N      [dim]N вариаций одного профиля[/dim]")
    console.print("  [bold]3[/bold] — Все группы × N       [dim]N студентов из каждой группы, сравнение + кластеры[/dim]")
    console.print("  [bold]4[/bold] — По одному из каждой  [dim]быстрое сравнение 4 профилей[/dim]")
    console.print("  [bold]5[/bold] — Прогрев кластеров    [dim]предсимуляция для инициализации кластеров[/dim]")
    console.print("  [bold]6[/bold] — A/B: LinUCB          [dim]идеальный эксперимент, ENABLE_CONTROL_GROUP=false[/dim]")
    console.print("  [bold]7[/bold] — A/B: Baseline        [dim]идеальный эксперимент, ENABLE_CONTROL_GROUP=true[/dim]")
    console.print("  [bold]8[/bold] — A/B: Сравнение       [dim]загружает последние summary и показывает таблицу[/dim]")
    console.print("  [bold]9[/bold] — Класс: единый план  [dim]4 типа × фиксированный класс + target KC → сравнение плана[/dim]")
    mode = _safe_input("  Режим (1–9, Enter=1): ").strip()

    # Дефолтное число заданий зависит от режима
    default_tasks = 250 if mode in ("6", "7", "9") else 80
    n_str = _safe_input(f"  Число заданий на студента (Enter={default_tasks}): ").strip()
    try:
        max_tasks = max(10, min(1000, int(n_str) if n_str else default_tasks))
    except ValueError:
        max_tasks = default_tasks

    if mode == "8":
        show_ab_comparison()
        return

    if mode == "6" or mode == "7":
        # A/B-эксперимент
        variant_slug = "linucb" if mode == "6" else "baseline"
        variant_name = "LinUCB" if mode == "6" else "Baseline (heuristic)"
        env_required = "ENABLE_CONTROL_GROUP=false" if mode == "6" else "ENABLE_CONTROL_GROUP=true"
        console.print()
        console.print(Panel(
            f"[bold]A/B-симуляция: {variant_name}[/bold]\n\n"
            f"  Убедись что retrieval-сервис запущен с [bold]{env_required}[/bold]\n"
            f"  (перезапусти если нужно: make restart-retrieval)\n\n"
            "  Профили: все 4 группы × разные классы (5/7/8/9)\n"
            "  Метрики: точность, MAE, освоено KC, time-to-mastery, KC прогресс, кластеры\n\n"
            "  [dim]⚠ Используй одинаковый RNG seed для обоих прогонов (LinUCB и Baseline)[/dim]",
            border_style="cyan", padding=(0, 2),
        ))

        seed_str = _safe_input("  RNG seed (Enter=42): ").strip()
        try:
            rng_seed = int(seed_str) if seed_str else 42
        except ValueError:
            rng_seed = 42

        n_str2 = _safe_input("  Студентов из каждой группы (Enter=30): ").strip()
        try:
            n_per_group = max(1, min(100, int(n_str2) if n_str2 else 30))
        except ValueError:
            n_per_group = 30

        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_tag = f"ab_{variant_slug}_{ts_tag}"

        base_rng = random.Random(rng_seed)
        profiles: list[StudentProfile] = []
        for base in PRESET_PROFILES.values():
            profiles.extend(_sample_profile_ab(base, i, base_rng) for i in range(n_per_group))

        total = len(profiles)
        console.print(
            f"\n[dim]Запускаю {total} студентов "
            f"({n_per_group} × 4 группы × классы {GRADE_POOL}) × {max_tasks} заданий...[/dim]\n"
            f"[dim]session_tag: {session_tag}[/dim]\n"
        )
        results = run_population(client, profiles, max_tasks, session_tag=session_tag)
        show_population_results(results, group_name=f"A/B: {variant_name}", grouped=True)
        show_cluster_alignment(results)
        save_session_summary(results, variant_slug, session_tag, rng_seed, max_tasks)
        return

    if mode == "5":
        # Прогрев кластеров
        n_str2 = _safe_input("  Студентов из каждой группы для прогрева (Enter=20): ").strip()
        try:
            n_warmup = max(1, min(50, int(n_str2) if n_str2 else 20))
        except ValueError:
            n_warmup = 20
        run_warmup(client, n_per_group=n_warmup, max_tasks=max_tasks)
        return

    elif mode == "4":
        # По одному из каждой группы
        console.print(f"\n[dim]Запускаю по 1 студенту из каждой группы × {max_tasks} заданий...[/dim]\n")
        profiles = list(PRESET_PROFILES.values())
        results = run_population(client, profiles, max_tasks)
        show_population_results(results)

    elif mode == "3":
        # N студентов из каждой группы
        n_str2 = _safe_input("  Сколько студентов из каждой группы (Enter=5): ").strip()
        try:
            n_per_group = max(1, min(100, int(n_str2) if n_str2 else 5))
        except ValueError:
            n_per_group = 5

        base_rng = random.Random()
        profiles: list[StudentProfile] = []
        for base in PRESET_PROFILES.values():
            profiles.extend(_sample_profile(base, i, base_rng) for i in range(n_per_group))

        total = len(profiles)
        console.print(
            f"\n[dim]Запускаю {total} студентов "
            f"({n_per_group} × 4 группы) × {max_tasks} заданий...[/dim]\n"
        )
        results = run_population(client, profiles, max_tasks)
        show_population_results(results, grouped=True)
        show_cluster_alignment(results)

    elif mode == "2":
        # N студентов из одного профиля
        console.print()
        base_profile = _pick_profile("  Базовый профиль (1–4, Enter=1): ")

        n_str2 = _safe_input("  Сколько студентов (Enter=5): ").strip()
        try:
            n_students = max(1, min(20, int(n_str2) if n_str2 else 5))
        except ValueError:
            n_students = 5

        console.print(
            f"\n[dim]Запускаю {n_students} студентов из группы "
            f"[bold]{base_profile.name}[/bold] × {max_tasks} заданий...[/dim]\n"
        )

        base_rng = random.Random()
        profiles = [_sample_profile(base_profile, i, base_rng) for i in range(n_students)]
        results = run_population(client, profiles, max_tasks)
        show_population_results(results, group_name=base_profile.name)

    elif mode == "9":
        # Класс: единый план для 4 типов студентов
        import dataclasses

        grade_str = _safe_input("  Класс (Enter=8): ").strip()
        try:
            grade = int(grade_str) if grade_str else 8
        except ValueError:
            grade = 8

        target_kc = _safe_input("  Целевая KC (Enter=kc_irrational_eq): ").strip() or "kc_irrational_eq"

        thresh_str = _safe_input("  mastery_threshold (Enter=0.85): ").strip()
        try:
            mastery_threshold = float(thresh_str) if thresh_str else 0.85
        except ValueError:
            mastery_threshold = 0.85

        plan_config = {
            "mode": "target_mastery",
            "target_kc_id": target_kc,
            "mastery_threshold": mastery_threshold,
        }

        profiles = [
            dataclasses.replace(base, grade=grade, name=f"{name} (кл.{grade})")
            for name, base in PRESET_PROFILES.items()
        ]

        console.print(
            f"\n[dim]Класс {grade} · target: {target_kc} · threshold: {mastery_threshold}[/dim]\n"
            f"[dim]Запускаю {len(profiles)} студентов × {max_tasks} заданий...[/dim]\n"
        )

        results: list[SimResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            for profile in profiles:
                sim_task = progress.add_task(f"[cyan]{profile.name}[/cyan]", total=max_tasks)

                def on_step(step: SimStep, tm: dict, vm: dict, _t=sim_task) -> None:
                    progress.update(_t, completed=step.task_num)

                result = run_single(client, profile, max_tasks, on_step=on_step, plan_config=plan_config)
                results.append(result)

        show_population_results(results)

        # Детальная план-таблица
        t2 = Table(title="Прогресс по плану", box=box.SIMPLE, header_style="bold dim")
        t2.add_column("Студент", style="bold", min_width=22)
        t2.add_column("Шагов", justify="right")
        t2.add_column("Выполнено", justify="right")
        t2.add_column("Освоено (план true/vis)", justify="right")
        t2.add_column("Фокус", justify="right")
        t2.add_column("Done@задание", justify="right")

        for r in results:
            total_steps = len(r.plan_kc_ids)
            if total_steps == 0:
                t2.add_row(r.profile.name, "—", "нет плана", "—", "—", "—")
                continue
            done_color = "green" if r.plan_steps_completed == total_steps else "yellow"
            focus_color = "green" if r.plan_focus_rate > 0.5 else ("yellow" if r.plan_focus_rate > 0.2 else "red")
            pm_t = r.plan_mastered_true
            pm_v = r.plan_mastered_visible
            pm_color = "green" if pm_t >= total_steps * 0.7 else ("yellow" if pm_t > 0 else "red")
            t2.add_row(
                r.profile.name,
                str(total_steps),
                f"[{done_color}]{r.plan_steps_completed}/{total_steps}[/{done_color}]",
                f"[{pm_color}]{pm_t}[/{pm_color}]/{pm_v}",
                f"[{focus_color}]{r.plan_focus_rate*100:.0f}%[/{focus_color}]",
                str(r.plan_completion_task) if r.plan_completion_task else "—",
            )
        console.print(t2)

    else:
        # Один студент — детальный отчёт
        console.print()
        profile = _pick_profile()
        console.print(f"\n[dim]Симулирую {max_tasks} заданий для [bold]{profile.name}[/bold]...[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            sim_task = progress.add_task(
                f"[cyan]{profile.name}[/cyan] true→visible", total=max_tasks
            )

            def on_step(step: SimStep, tm: dict, vm: dict) -> None:
                progress.update(sim_task, completed=step.task_num)

            result = run_single(client, profile, max_tasks, on_step=on_step)

        show_single_result(result)
