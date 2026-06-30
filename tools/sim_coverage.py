#!/usr/bin/env python3
"""
Симуляция coverage-плана для разных типов студентов.

Создаёт студента, строит coverage-план, прогоняет задания и отслеживает
сколько KC из плана реально освоено (true mastery) vs по мнению системы (visible).

Запуск:
    python tools/sim_coverage.py
    python tools/sim_coverage.py --grades 6 8 10 --tasks 300
    python tools/sim_coverage.py --variants count mass frontier --n 3
    python tools/sim_coverage.py --profile fast_learner --grade 8 --variant frontier --tasks 400 --detail
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.simulation import (
    MASTERY_THRESHOLD, PRESET_PROFILES, StudentProfile,
    simulate_answer, _true_mastery_update, _smooth_update, _sample_profile,
    console,
)

# ─── Настройки ───────────────────────────────────────────────────────────────

GATEWAY_URL = "http://127.0.0.1:8005"
PROFILE_URL = "http://127.0.0.1:8001"
MACRO_URL   = "http://127.0.0.1:8006"
TIMEOUT     = 10.0


# ─── API-хелперы ─────────────────────────────────────────────────────────────

def _create_student(http: httpx.Client, grade: int) -> dict:
    r = http.post(f"{GATEWAY_URL}/students", json={"grade": grade}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _get_mastery(http: httpx.Client, student_id: str) -> list[dict]:
    r = http.get(f"{PROFILE_URL}/students/{student_id}/mastery", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _create_plan(http: httpx.Client, student_id: str, variant: str, task_budget: int) -> dict:
    r = http.post(f"{MACRO_URL}/plans", json={
        "student_id": student_id,
        "mode": "coverage",
        "coverage_variant": variant,
        "task_budget": task_budget,
    }, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _get_plan(http: httpx.Client, plan_id: str) -> dict:
    r = http.get(f"{MACRO_URL}/plans/{plan_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _advance_plan(http: httpx.Client, plan_id: str) -> None:
    http.post(f"{MACRO_URL}/plans/{plan_id}/advance", timeout=TIMEOUT)


def _next_task(http: httpx.Client, student_id: str, last_subject: Optional[str] = None) -> dict:
    params = {}
    if last_subject:
        params["last_subject"] = last_subject
    r = http.get(f"{GATEWAY_URL}/students/{student_id}/next-task", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _submit(http: httpx.Client, student_id: str, rec: dict, score: float) -> dict:
    part = (rec.get("task", {}).get("parts") or [{}])[0]
    r = http.post(f"{GATEWAY_URL}/sessions/{student_id}/answer", json={
        "task_id": rec["task_id"],
        "part_id": part.get("id", "main"),
        "score": score,
        "hints_used": 0,
        "primary_kcs": [rec["kc_id"]],
        "recommendation_source": rec.get("recommendation_source"),
    }, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ─── Результат одной симуляции ────────────────────────────────────────────────

@dataclass
class CoverageResult:
    profile_name: str
    grade: int
    variant: str
    student_id: str
    plan_kcs: list[str]          # KC в плане (по порядку)
    tasks_done: int = 0
    plan_mastered_true: int = 0
    plan_mastered_visible: int = 0
    accuracy: float = 0.0
    plan_focus_rate: float = 0.0  # доля заданий на KC из плана
    steps_completed: int = 0
    # per-KC детали
    kc_tasks: dict[str, int] = dataclasses.field(default_factory=dict)
    kc_true_final: dict[str, float] = dataclasses.field(default_factory=dict)
    kc_vis_final: dict[str, float] = dataclasses.field(default_factory=dict)
    plan_kc_vis_at_creation: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def plan_size(self) -> int:
        return len(self.plan_kcs)

    @property
    def coverage_pct(self) -> float:
        return self.plan_mastered_true / self.plan_size * 100 if self.plan_size else 0.0


# ─── Ядро симуляции ──────────────────────────────────────────────────────────

def run_coverage_single(
    http: httpx.Client,
    profile: StudentProfile,
    variant: str,
    task_budget: int,
    max_tasks: int,
    rng: random.Random,
    on_step=None,
) -> CoverageResult:
    # 1. Создать студента
    student_data = _create_student(http, profile.grade)
    student_id = student_data["student_id"]

    # 2. Инициализировать true mastery из cold_start
    visible_mastery: dict[str, float] = {}
    for rec in _get_mastery(http, student_id):
        visible_mastery[rec["kc_id"]] = rec.get("probability", 0.0)

    true_mastery: dict[str, float] = {
        kc_id: (profile.initial_true_mastery if vis <= 0.50 else vis)
        for kc_id, vis in visible_mastery.items()
    }
    true_mastery.update(profile.specific_kcs)
    init_true = dict(true_mastery)

    # 3. Создать coverage план
    plan_id: Optional[str] = None
    plan_kcs: list[str] = []
    plan_kc_vis_at_creation: dict[str, float] = {}
    try:
        plan_resp = _create_plan(http, student_id, variant, task_budget)
        plan_id = plan_resp.get("plan_id")
        if plan_id:
            plan_data = _get_plan(http, plan_id)
            plan_kcs = [s["kc_id"] for s in plan_data.get("steps", [])]
            plan_kc_vis_at_creation = {kc: visible_mastery.get(kc, 0.0) for kc in plan_kcs}
    except Exception as e:
        console.print(f"  [yellow]Не удалось создать план: {e}[/yellow]")

    plan_kc_set = set(plan_kcs)
    plan_step_idx = 0

    result = CoverageResult(
        profile_name=profile.name,
        grade=profile.grade,
        variant=variant,
        student_id=student_id,
        plan_kcs=plan_kcs,
    )

    correct_count = 0
    plan_task_count = 0
    last_subject: Optional[str] = None

    for task_num in range(1, max_tasks + 1):
        # Получить рекомендацию
        try:
            rec = _next_task(http, student_id, last_subject)
        except Exception:
            break

        kc_id = rec["kc_id"]
        part = (rec.get("task", {}).get("parts") or [{}])[0]
        irt_diff = part.get("irt_difficulty", 0.5)
        task_type = part.get("task_type", "procedural")
        last_subject = rec.get("subject")

        # Симулировать ответ
        true_m = true_mastery.get(kc_id, profile.initial_true_mastery)
        type_diff_offset = profile.type_difficulty_mod.get(task_type, 0.0)
        eff_diff = min(0.99, max(0.01, irt_diff + type_diff_offset))
        score = simulate_answer(true_m, eff_diff, profile.p_slip, profile.p_guess,
                                profile.careless_error_rate, rng)

        # Отправить в систему
        try:
            resp = _submit(http, student_id, rec, score)
        except Exception:
            break

        # Обновить visible mastery
        updated = resp.get("mastery_update", {}).get("updated_mastery", {})
        for kc, v in updated.items():
            visible_mastery[kc] = v

        # Трекинг
        if score >= 0.9:
            correct_count += 1
        if kc_id in plan_kc_set:
            plan_task_count += 1

        result.kc_tasks[kc_id] = result.kc_tasks.get(kc_id, 0) + 1

        # Обновить true mastery
        type_lr_mult = profile.type_learning_mod.get(task_type, 1.0)
        new_true = _true_mastery_update(true_m, score, profile.growth_scale * type_lr_mult)
        true_mastery[kc_id] = new_true

        # Продвинуть шаг плана если KC освоена по visible mastery
        if plan_id and plan_kcs and plan_step_idx < len(plan_kcs):
            current_kc = plan_kcs[plan_step_idx]
            if visible_mastery.get(current_kc, 0.0) >= MASTERY_THRESHOLD:
                try:
                    _advance_plan(http, plan_id)
                except Exception:
                    pass
                plan_step_idx += 1
                result.steps_completed = plan_step_idx

        if on_step:
            on_step(task_num)

    result.tasks_done = max_tasks if plan_step_idx == 0 else task_num
    result.tasks_done = task_num
    result.accuracy = correct_count / task_num if task_num else 0.0
    result.plan_focus_rate = plan_task_count / task_num if task_num else 0.0
    result.steps_completed = plan_step_idx

    # Финальные метрики по KC плана
    for kc in plan_kcs:
        ft = true_mastery.get(kc, 0.0)
        fv = visible_mastery.get(kc, 0.0)
        result.kc_true_final[kc] = ft
        result.kc_vis_final[kc] = fv
        if ft >= MASTERY_THRESHOLD and init_true.get(kc, 0.0) < MASTERY_THRESHOLD:
            result.plan_mastered_true += 1
        if fv >= MASTERY_THRESHOLD:
            result.plan_mastered_visible += 1

    result.plan_kc_vis_at_creation = plan_kc_vis_at_creation
    _save_log(result, plan_kc_vis_at_creation)
    return result


LOG_DIR = Path(__file__).parent.parent / "logs" / "sim_coverage"


def _save_log(result: CoverageResult, plan_kc_vis_at_creation: dict[str, float]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{ts}_{result.profile_name}_{result.variant}.json"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "profile": result.profile_name,
        "grade": result.grade,
        "variant": result.variant,
        "student_id": result.student_id,
        "tasks_done": result.tasks_done,
        "accuracy": round(result.accuracy, 4),
        "plan_focus_rate": round(result.plan_focus_rate, 4),
        "plan_mastered_true": result.plan_mastered_true,
        "plan_mastered_visible": result.plan_mastered_visible,
        "coverage_pct": round(result.coverage_pct, 1),
        "steps_completed": result.steps_completed,
        "plan_kcs": result.plan_kcs,
        # Диагностика: visible mastery KC плана в момент создания плана
        "plan_kc_vis_at_creation": {k: round(v, 3) for k, v in plan_kc_vis_at_creation.items()},
        # Финальный mastery по KC плана
        "plan_kc_final": {
            kc: {
                "true": round(result.kc_true_final.get(kc, 0.0), 3),
                "vis": round(result.kc_vis_final.get(kc, 0.0), 3),
                "tasks": result.kc_tasks.get(kc, 0),
            }
            for kc in result.plan_kcs
        },
        # Все KC которые реально получали задания
        "all_kc_tasks": {k: v for k, v in sorted(result.kc_tasks.items(), key=lambda x: -x[1])},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ─── Отображение ─────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 8) -> str:
    filled = round(v * w)
    color = "green" if v >= MASTERY_THRESHOLD else ("yellow" if v >= 0.40 else "red")
    return f"[{color}]{'█' * filled}{'░' * (w - filled)}[/{color}] {v:.2f}"


def show_detail(result: CoverageResult) -> None:
    """Детальная таблица по KC плана для одного прогона."""
    t = Table(
        title=f"[bold]{result.profile_name}[/bold] · кл.{result.grade} · variant={result.variant}",
        box=box.ROUNDED, header_style="bold dim",
    )
    t.add_column("KC", min_width=28)
    t.add_column("True", width=14)
    t.add_column("Visible", width=14)
    t.add_column("|Δ|", justify="right", width=6)
    t.add_column("Задания", justify="right", width=8)

    for kc in result.plan_kcs:
        ft = result.kc_true_final.get(kc, 0.0)
        fv = result.kc_vis_final.get(kc, 0.0)
        delta = abs(ft - fv)
        d_color = "red" if delta > 0.25 else ("yellow" if delta > 0.10 else "green")
        t.add_row(
            kc, _bar(ft), _bar(fv),
            f"[{d_color}]{delta:.2f}[/{d_color}]",
            str(result.kc_tasks.get(kc, 0)),
        )

    console.print(t)
    console.print(
        f"  Задания: {result.tasks_done}  "
        f"Точность: {result.accuracy*100:.0f}%  "
        f"Фокус на плане: {result.plan_focus_rate*100:.0f}%  "
        f"Шагов выполнено: {result.steps_completed}/{result.plan_size}"
    )


def show_comparison(results: list[CoverageResult], max_tasks: int) -> None:
    """Сравнительная таблица по всем прогонам."""
    console.print()
    t = Table(
        title=f"Coverage-план · сравнение ({max_tasks} заданий/студент)",
        box=box.ROUNDED, header_style="bold dim",
    )
    t.add_column("Профиль", style="bold", min_width=18)
    t.add_column("Класс", justify="center", width=6)
    t.add_column("Вариант", width=10)
    t.add_column("KC в плане", justify="right", width=10)
    t.add_column("Освоено true", justify="right", width=12)
    t.add_column("Освоено vis", justify="right", width=11)
    t.add_column("Покрытие %", justify="right", width=10)
    t.add_column("Точность", justify="right", width=9)
    t.add_column("Фокус", justify="right", width=7)

    prev_key = None
    for r in results:
        key = (r.profile_name.rsplit("_", 1)[0], r.grade, r.variant)
        if prev_key and (key[0] != prev_key[0] or key[1] != prev_key[1]):
            t.add_section()
        prev_key = key

        cov_color = "green" if r.coverage_pct >= 70 else ("yellow" if r.coverage_pct >= 40 else "red")
        acc_color = "green" if r.accuracy >= 0.65 else ("yellow" if r.accuracy >= 0.45 else "red")
        t.add_row(
            r.profile_name,
            str(r.grade),
            r.variant,
            str(r.plan_size),
            f"[{cov_color}]{r.plan_mastered_true}[/{cov_color}]",
            str(r.plan_mastered_visible),
            f"[{cov_color}]{r.coverage_pct:.0f}%[/{cov_color}]",
            f"[{acc_color}]{r.accuracy*100:.0f}%[/{acc_color}]",
            f"{r.plan_focus_rate*100:.0f}%",
        )

    console.print(t)

    # Агрегация по варианту
    from itertools import groupby
    by_variant: dict[str, list[CoverageResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    if len(by_variant) > 1:
        console.print("\n  [bold dim]Среднее по варианту:[/bold dim]")
        for variant, vr in sorted(by_variant.items()):
            avg_cov = sum(r.coverage_pct for r in vr) / len(vr)
            avg_acc = sum(r.accuracy for r in vr) / len(vr)
            avg_focus = sum(r.plan_focus_rate for r in vr) / len(vr)
            cov_c = "green" if avg_cov >= 70 else ("yellow" if avg_cov >= 40 else "red")
            console.print(
                f"  [cyan]{variant}[/cyan] (n={len(vr)}): "
                f"покрытие [{cov_c}]{avg_cov:.0f}%[/{cov_c}]  "
                f"точность {avg_acc*100:.0f}%  "
                f"фокус {avg_focus*100:.0f}%"
            )

    console.print()


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-план симуляция")
    parser.add_argument("--grades",   type=int, nargs="+", default=[8],
                        help="Классы студентов (по умолчанию: 8)")
    parser.add_argument("--variants", nargs="+",
                        choices=["count", "mass", "frontier"],
                        default=["count", "mass", "frontier"],
                        help="Coverage-варианты для сравнения")
    parser.add_argument("--profiles", nargs="+",
                        choices=list(PRESET_PROFILES.keys()),
                        default=list(PRESET_PROFILES.keys()),
                        help="Профили студентов")
    parser.add_argument("--tasks",   type=int, default=150,
                        help="Заданий на студента (по умолчанию: 150)")
    parser.add_argument("--budget",  type=int, default=200,
                        help="task_budget для построения плана (по умолчанию: 200)")
    parser.add_argument("--n",       type=int, default=1,
                        help="Студентов на каждую комбинацию профиль×класс×вариант")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--detail",  action="store_true",
                        help="Показать детальную таблицу по KC для каждого прогона")
    args = parser.parse_args()

    base_rng = random.Random(args.seed)

    # Проверить доступность сервисов
    with httpx.Client(timeout=3.0) as http:
        for url, name in [(GATEWAY_URL, "gateway"), (MACRO_URL, "macro"), (PROFILE_URL, "profile")]:
            try:
                http.get(f"{url}/health").raise_for_status()
            except Exception as e:
                console.print(f"[red]{name} недоступен: {e}[/red]")
                console.print("[dim]Запусти: make dev[/dim]")
                sys.exit(1)

    total = len(args.profiles) * len(args.grades) * len(args.variants) * args.n
    console.print()
    console.print(Panel(
        f"[bold]Coverage-план симуляция[/bold]\n\n"
        f"  Профили:  {', '.join(args.profiles)}\n"
        f"  Классы:   {args.grades}\n"
        f"  Варианты: {args.variants}\n"
        f"  Задания:  {args.tasks} на студента · budget={args.budget}\n"
        f"  Всего студентов: [bold]{total}[/bold]",
        border_style="cyan", padding=(0, 2),
    ))

    all_results: list[CoverageResult] = []

    with httpx.Client(timeout=TIMEOUT) as http:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            prog_task = progress.add_task("симуляция", total=total * args.tasks)
            done_students = 0

            for profile_name in args.profiles:
                base_profile = PRESET_PROFILES[profile_name]
                for grade in args.grades:
                    grade_profile = dataclasses.replace(base_profile, grade=grade)
                    for variant in args.variants:
                        for i in range(args.n):
                            rng = random.Random(base_rng.randint(0, 2**32))
                            profile = _sample_profile(grade_profile, i, rng) if args.n > 1 else grade_profile

                            completed_tasks = [0]

                            def on_step(t, _c=completed_tasks, _pg=progress, _pt=prog_task):
                                _c[0] = t
                                _pg.update(_pt, completed=done_students * args.tasks + t)

                            progress.update(
                                prog_task,
                                description=f"[cyan]{profile_name}[/cyan] кл.{grade} {variant}",
                            )

                            try:
                                result = run_coverage_single(
                                    http, profile, variant, args.budget,
                                    args.tasks, rng, on_step=on_step,
                                )
                                all_results.append(result)
                                if args.detail:
                                    show_detail(result)
                            except Exception as e:
                                console.print(
                                    f"\n  [red]Ошибка: {profile_name} кл.{grade} {variant}: {e}[/red]"
                                )

                            done_students += 1
                            progress.update(prog_task, completed=done_students * args.tasks)

    if all_results:
        show_comparison(all_results, args.tasks)


if __name__ == "__main__":
    main()
