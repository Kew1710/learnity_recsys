"""
Тест построения coverage-плана для студентов разных классов.

Запуск (сервисы должны быть запущены: make dev):
    python tools/test_coverage_plan.py
    python tools/test_coverage_plan.py --grades 6 8 10 --variant frontier
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

GATEWAY_URL = "http://127.0.0.1:8005"
MACRO_URL   = "http://127.0.0.1:8006"
PROFILE_URL = "http://127.0.0.1:8001"
TIMEOUT     = 15.0

console = Console()


def create_student(http: httpx.Client, grade: int) -> str:
    resp = http.post(f"{GATEWAY_URL}/students", json={"grade": grade}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["student_id"]


def create_coverage_plan(http: httpx.Client, student_id: str, variant: str, task_budget: int) -> dict:
    resp = http.post(f"{MACRO_URL}/plans", json={
        "student_id": student_id,
        "mode": "coverage",
        "coverage_variant": variant,
        "task_budget": task_budget,
    }, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_plan(http: httpx.Client, plan_id: str) -> dict:
    resp = http.get(f"{MACRO_URL}/plans/{plan_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_mastery(http: httpx.Client, student_id: str) -> dict:
    resp = http.get(f"{PROFILE_URL}/students/{student_id}/mastery", timeout=TIMEOUT)
    resp.raise_for_status()
    return {r["kc_id"]: r["probability"] for r in resp.json()}


def show_plan(grade: int, student_id: str, variant: str, plan: dict, mastery: dict) -> None:
    steps = plan.get("steps", [])

    table = Table(box=box.SIMPLE, header_style="bold dim", show_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("KC", min_width=28)
    table.add_column("Mastery", justify="right", width=8)
    table.add_column("Budget", justify="right", width=7)
    table.add_column("Mode", width=12)
    table.add_column("Статус", width=12)

    for i, step in enumerate(steps, 1):
        kc = step["kc_id"]
        m = mastery.get(kc, 0.0)
        m_color = "green" if m >= 0.75 else ("yellow" if m >= 0.4 else "red")
        status_color = "green" if step["status"] == "completed" else ("cyan" if step["status"] == "in_progress" else "dim")
        table.add_row(
            str(i),
            kc,
            f"[{m_color}]{m:.2f}[/{m_color}]",
            str(step.get("tasks_budget", "—")),
            step.get("difficulty_mode", "—"),
            f"[{status_color}]{step['status']}[/{status_color}]",
        )

    console.print(Panel(
        table,
        title=f"[bold]Класс {grade} · variant={variant} · {len(steps)} шагов[/bold]",
        subtitle=f"[dim]student={student_id[:8]}...[/dim]",
        border_style="blue",
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--variant", choices=["count", "mass", "frontier"], default="count")
    parser.add_argument("--variants", action="store_true", help="Показать все 3 варианта для каждого класса")
    parser.add_argument("--budget", type=int, default=200)
    args = parser.parse_args()

    variants = ["count", "mass", "frontier"] if args.variants else [args.variant]

    with httpx.Client(timeout=TIMEOUT) as http:
        # Проверяем доступность сервисов
        try:
            http.get(f"{GATEWAY_URL}/health", timeout=3).raise_for_status()
            http.get(f"{MACRO_URL}/health", timeout=3).raise_for_status()
        except Exception as e:
            console.print(f"[red]Сервисы недоступны: {e}[/red]")
            console.print("[dim]Запусти: make dev[/dim]")
            sys.exit(1)

        for grade in args.grades:
            console.print(f"\n[bold cyan]── Класс {grade} ──[/bold cyan]")

            try:
                student_id = create_student(http, grade)
                console.print(f"  [dim]Студент создан: {student_id[:8]}...[/dim]")
            except Exception as e:
                console.print(f"  [red]Ошибка создания студента: {e}[/red]")
                continue

            mastery = get_mastery(http, student_id)

            for variant in variants:
                try:
                    plan_resp = create_coverage_plan(http, student_id, variant, args.budget)
                    plan = get_plan(http, plan_resp["plan_id"])
                    show_plan(grade, student_id, variant, plan, mastery)
                except Exception as e:
                    console.print(f"  [red]Ошибка построения плана (variant={variant}): {e}[/red]")


if __name__ == "__main__":
    main()
