"""
Запускает mode 9 (groupsim) без интерактивных промптов.
Использование: python tools/run_groupsim.py [grade] [target_kc] [max_tasks]
"""
import sys
import dataclasses
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box
from rich.table import Table

sys.path.insert(0, ".")
from tools.simulation import (
    PRESET_PROFILES, run_single, show_population_results, SimStep
)

console = Console()

GATEWAY_URL = "http://127.0.0.1:8005"

grade = int(sys.argv[1]) if len(sys.argv) > 1 else 8
target_kc = sys.argv[2] if len(sys.argv) > 2 else "kc_irrational_eq"
max_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else 250
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

results = []
with httpx.Client(base_url=GATEWAY_URL, timeout=30) as client:
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

            def on_step(step: SimStep, tm, vm, _t=sim_task) -> None:
                progress.update(_t, completed=step.task_num)

            result = run_single(client, profile, max_tasks, on_step=on_step, plan_config=plan_config)
            results.append(result)

show_population_results(results)

t2 = Table(title="Прогресс по плану", box=box.SIMPLE, header_style="bold dim")
t2.add_column("Студент", style="bold", min_width=22)
t2.add_column("Шагов", justify="right")
t2.add_column("Выполнено", justify="right")
t2.add_column("Фокус", justify="right")
t2.add_column("Done@задание", justify="right")

for r in results:
    total_steps = len(r.plan_kc_ids)
    if total_steps == 0:
        t2.add_row(r.profile.name, "—", "нет плана", "—", "—")
        continue
    done_color = "green" if r.plan_steps_completed == total_steps else "yellow"
    focus_color = "green" if r.plan_focus_rate > 0.5 else ("yellow" if r.plan_focus_rate > 0.2 else "red")
    t2.add_row(
        r.profile.name,
        str(total_steps),
        f"[{done_color}]{r.plan_steps_completed}/{total_steps}[/{done_color}]",
        f"[{focus_color}]{r.plan_focus_rate*100:.0f}%[/{focus_color}]",
        str(r.plan_completion_task) if r.plan_completion_task else "—",
    )
console.print(t2)
