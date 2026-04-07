#!/usr/bin/env python3
"""
Консольный REPL для ручного тестирования рекомендательной системы.

Запуск:
    python tools/play.py
    make play

Требует запущенных сервисов:
    make up   # инфраструктура
    make dev  # сервисы
"""

import sys
import os
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataclasses import dataclass, field
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.columns import Columns

# ─── Настройки ───────────────────────────────────────────────────────────────

GATEWAY_URL = "http://127.0.0.1:8005"
PROFILE_URL = "http://127.0.0.1:8001"
TIMEOUT = 8.0

SUBJECT_COLORS = {
    "arithmetic":  "cyan",
    "algebra":     "green",
    "geometry":    "yellow",
    "statistics":  "magenta",
}
SUBJECT_RU = {
    "arithmetic":  "Арифметика",
    "algebra":     "Алгебра",
    "geometry":    "Геометрия",
    "statistics":  "Статистика",
}

console = Console()


# ─── Данные сессии ────────────────────────────────────────────────────────────

@dataclass
class SessionStats:
    tasks_done: int = 0
    correct: int = 0
    partial: int = 0
    wrong: int = 0
    zpd_count: int = 0
    explore_count: int = 0
    subjects_seen: list[str] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        return self.correct / self.tasks_done * 100 if self.tasks_done else 0.0


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def difficulty_bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    bar = "█" * filled + "░" * (width - filled)
    if value < 0.35:
        label = "лёгкое"
    elif value < 0.65:
        label = "среднее"
    else:
        label = "сложное"
    return f"{bar} {value:.2f}  [{label}]"


def mastery_bar(value: float, width: int = 8) -> str:
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def source_label(source: str) -> str:
    if source == "zpd":
        return "[green]ZPD (exploitation)[/green]"
    elif source == "exploration":
        return "[yellow]Exploration (случайное)[/yellow]"
    return source


def check_services() -> bool:
    """Быстрая проверка доступности Gateway."""
    try:
        r = httpx.get(f"{GATEWAY_URL}/health", timeout=3.0, trust_env=False)
        return r.status_code == 200
    except Exception:
        return False


# ─── API-вызовы ───────────────────────────────────────────────────────────────

def api_create_student(client: httpx.Client, grade: int) -> dict:
    r = client.post(f"{GATEWAY_URL}/students", json={"grade": grade}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_first_task(client: httpx.Client, student_id: str, last_subject: Optional[str]) -> dict:
    params = {}
    if last_subject:
        params["last_subject"] = last_subject
    r = client.get(f"{GATEWAY_URL}/students/{student_id}/next-task", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_submit(
    client: httpx.Client,
    student_id: str,
    task_id: str,
    part_id: str,
    score: float,
    primary_kcs: list[str],
    half_life_days: float,
    recommendation_source: str,
    last_subject: Optional[str],
) -> dict:
    payload = {
        "task_id": task_id,
        "part_id": part_id,
        "score": score,
        "primary_kcs": primary_kcs,
        "half_life_days": half_life_days,
        "recommendation_source": recommendation_source,
        "last_subject": last_subject,
    }
    r = client.post(f"{GATEWAY_URL}/sessions/{student_id}/answer", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_full_mastery(client: httpx.Client, student_id: str) -> list[dict]:
    r = client.get(f"{PROFILE_URL}/students/{student_id}/mastery", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ─── Отображение ──────────────────────────────────────────────────────────────

def show_banner():
    console.print()
    console.print(Panel(
        "[bold white]Learnity — ручное тестирование рекомендаций[/bold white]\n"
        "[dim]Используй реальные задания или просто укажи скор для теста системы[/dim]",
        border_style="blue",
        padding=(0, 2),
    ))
    console.print()


def show_task_card(rec: dict, task_num: int):
    part = rec["task"]["parts"][0]
    subject = rec["subject"]
    color = SUBJECT_COLORS.get(subject, "white")
    subject_ru = SUBJECT_RU.get(subject, subject.capitalize())

    # Заголовок
    header = Text()
    header.append(f"  Задание #{task_num}", style="bold white")
    header.append(f"  [{subject_ru}]", style=f"bold {color}")

    # Тело карточки
    lines = []
    lines.append(f"[dim]KC:[/dim]          {rec['kc_id']}")
    lines.append(f"[dim]Описание:[/dim]    {part['description']}")
    lines.append(f"[dim]Сложность:[/dim]   {difficulty_bar(part['irt_difficulty'] or 0.5)}")
    lines.append(f"[dim]Источник:[/dim]    {source_label(rec['recommendation_source'])}")
    lines.append(f"[dim]half_life:[/dim]   {rec['half_life_days']} дней")
    lines.append("")
    lines.append("[dim italic]Это синтетическое задание — реального текста нет.[/dim italic]")
    lines.append("[dim italic]Укажи скор вручную (1=верно, 0=неверно, 0.5=частично).[/dim italic]")

    console.print()
    console.rule(style=f"{color} dim")
    console.print(header)
    console.rule(style=f"{color} dim")
    for line in lines:
        console.print(f"  {line}")
    console.print()


def show_mastery_delta(mastery_update: dict, primary_kcs: list[str]):
    updated = mastery_update.get("updated_mastery", {})
    before = mastery_update.get("mastery_before", {})

    if not updated:
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("KC", style="dim")
    table.add_column("До", justify="right")
    table.add_column("После", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Прогресс", no_wrap=True)

    for kc_id in primary_kcs:
        if kc_id not in updated:
            continue
        b = before.get(kc_id, 0.0)
        a = updated[kc_id]
        delta = a - b
        delta_str = f"[green]+{delta:.3f}[/green]" if delta >= 0 else f"[red]{delta:.3f}[/red]"
        table.add_row(
            kc_id,
            f"{b:.3f}",
            f"{a:.3f}",
            delta_str,
            mastery_bar(a),
        )

    console.print(table)


def show_session_stats(stats: SessionStats):
    console.print()
    console.print(
        f"[dim]Сессия:[/dim]  "
        f"{stats.tasks_done} заданий  |  "
        f"[green]{stats.correct} верно[/green]  "
        f"[yellow]{stats.partial} частично[/yellow]  "
        f"[red]{stats.wrong} неверно[/red]  |  "
        f"Успех: [bold]{stats.score_pct:.0f}%[/bold]  |  "
        f"ZPD: {stats.zpd_count}  Explore: {stats.explore_count}"
    )


def show_full_mastery(client: httpx.Client, student_id: str):
    try:
        records = api_full_mastery(client, student_id)
    except Exception as e:
        console.print(f"[red]Ошибка при получении mastery: {e}[/red]")
        return

    if not records:
        console.print("[dim]Mastery пуст — ещё не было ответов[/dim]")
        return

    records.sort(key=lambda r: r["probability_effective"], reverse=True)

    table = Table(title="Mastery ученика", box=box.ROUNDED, show_header=True)
    table.add_column("KC", style="dim", max_width=35)
    table.add_column("P(знает)", justify="right")
    table.add_column("P(eff)", justify="right")
    table.add_column("Попыток", justify="right")
    table.add_column("Прогресс", no_wrap=True)

    for r in records[:30]:
        p = r["probability"]
        pe = r["probability_effective"]
        color = "green" if pe >= 0.7 else ("yellow" if pe >= 0.4 else "red")
        table.add_row(
            r["kc_id"],
            f"{p:.3f}",
            f"[{color}]{pe:.3f}[/{color}]",
            str(r["attempts_count"]),
            mastery_bar(pe),
        )

    if len(records) > 30:
        table.add_row("...", f"(ещё {len(records) - 30})", "", "", "")

    console.print()
    console.print(table)


def show_session_summary(stats: SessionStats, student_id: str):
    console.print()
    console.print(Panel(
        f"[bold]Итоги сессии[/bold]\n\n"
        f"  Заданий пройдено:   {stats.tasks_done}\n"
        f"  Правильно:          [green]{stats.correct}[/green] ({stats.score_pct:.0f}%)\n"
        f"  Частично:           [yellow]{stats.partial}[/yellow]\n"
        f"  Неправильно:        [red]{stats.wrong}[/red]\n\n"
        f"  ZPD (exploitation): {stats.zpd_count}\n"
        f"  Exploration:        {stats.explore_count}\n\n"
        f"  [dim]student_id: {student_id}[/dim]",
        border_style="blue",
        padding=(0, 2),
    ))


# ─── Главный цикл ─────────────────────────────────────────────────────────────

def get_score_input() -> Optional[float]:
    """
    Возвращает:
      float 0.0–1.0 — скор
      None          — специальная команда (уже обработана внутри)
      'quit'        — выйти
    """
    while True:
        console.print(
            "[bold]Ответ:[/bold]  "
            "[green]1[/green]=верно  "
            "[red]0[/red]=неверно  "
            "[yellow]0.5[/yellow]=частично  "
            "[dim]m[/dim]=mastery  "
            "[dim]s[/dim]=пропустить  "
            "[dim]q[/dim]=выйти"
        )
        raw = console.input("> ").strip().lower()

        if raw in ("q", "quit", "exit"):
            return "quit"
        if raw in ("m", "mastery"):
            return "mastery"
        if raw in ("s", "skip", ""):
            return "skip"
        if raw == "1":
            return 1.0
        if raw == "0":
            return 0.0
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return val
            console.print("[red]Введи число от 0 до 1[/red]")
        except ValueError:
            console.print("[red]Неизвестная команда. Введи 1, 0, 0.5 или команду.[/red]")


def run():
    show_banner()

    # ── Проверка сервисов ──
    console.print("[dim]Проверяю соединение с Gateway...[/dim]")
    if not check_services():
        console.print(
            "[red bold]Gateway не отвечает на http://localhost:8000[/red bold]\n"
            "[dim]Запусти: make up && make dev && make seed[/dim]"
        )
        sys.exit(1)
    console.print("[green]✓ Gateway доступен[/green]")
    console.print()

    # ── Выбор ученика ──
    console.print("Новый ученик или существующий?")
    console.print("  [bold]n[/bold] — создать нового")
    console.print("  [bold]<id>[/bold] — ввести существующий student_id")
    choice = console.input("> ").strip().lower()

    client = httpx.Client(trust_env=False)
    student_id: str = ""

    if choice == "n":
        while True:
            grade_str = console.input("Класс (5–11): ").strip()
            try:
                grade = int(grade_str)
                if 5 <= grade <= 11:
                    break
                console.print("[red]Класс должен быть от 5 до 11[/red]")
            except ValueError:
                console.print("[red]Введи число[/red]")

        try:
            result = api_create_student(client, grade)
            student_id = result["student_id"]
            console.print(f"[green]✓ Ученик создан:[/green] {student_id}  Класс: {grade}")
        except Exception as e:
            console.print(f"[red]Ошибка создания ученика: {e}[/red]")
            sys.exit(1)
    else:
        student_id = choice
        console.print(f"[dim]Используем student_id: {student_id}[/dim]")

    # ── Первое задание ──
    stats = SessionStats()
    last_subject: Optional[str] = None
    current_rec: Optional[dict] = None

    console.print("\n[dim]Получаю первое задание...[/dim]")
    try:
        current_rec = api_first_task(client, student_id, None)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Ошибка: {e.response.status_code} — {e.response.text}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Ошибка соединения: {e}[/red]")
        sys.exit(1)

    # ── Основной цикл ──
    while current_rec:
        task_num = stats.tasks_done + 1
        show_task_card(current_rec, task_num)

        part = current_rec["task"]["parts"][0]
        action = get_score_input()

        if action == "quit":
            break

        if action == "mastery":
            show_full_mastery(client, student_id)
            continue   # показываем то же задание снова

        if action == "skip":
            console.print("[dim]Задание пропущено[/dim]")
            # Получаем следующее без submit
            try:
                current_rec = api_first_task(client, student_id, current_rec["subject"])
                last_subject = current_rec["subject"]
            except Exception as e:
                console.print(f"[red]Ошибка: {e}[/red]")
                break
            continue

        score: float = action

        # Обновляем счётчики
        stats.tasks_done += 1
        if score >= 0.9:
            stats.correct += 1
            result_label = "[green]✓ Верно![/green]"
        elif score >= 0.4:
            stats.partial += 1
            result_label = "[yellow]△ Частично[/yellow]"
        else:
            stats.wrong += 1
            result_label = "[red]✗ Неверно[/red]"

        if current_rec["recommendation_source"] == "exploration":
            stats.explore_count += 1
        else:
            stats.zpd_count += 1
        stats.subjects_seen.append(current_rec["subject"])

        # Submit
        try:
            response = api_submit(
                client=client,
                student_id=student_id,
                task_id=str(current_rec["task"]["task_id"]),
                part_id=part["part_id"],
                score=score,
                primary_kcs=part["primary_kcs"],
                half_life_days=current_rec["half_life_days"],
                recommendation_source=current_rec["recommendation_source"],
                last_subject=last_subject,
            )
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Ошибка submit: {e.response.status_code}[/red]")
            break
        except Exception as e:
            console.print(f"[red]Ошибка соединения: {e}[/red]")
            break

        # Результат
        console.print(f"\n{result_label}  (скор: {score:.2f})")
        show_mastery_delta(response["mastery_update"], part["primary_kcs"])

        last_subject = current_rec["subject"]

        show_session_stats(stats)

        # Следующее задание из ответа
        next_rec = response.get("next_task")
        if not next_rec:
            console.print("[dim]Нет следующего задания (TaskBank исчерпан)[/dim]")
            break

        current_rec = next_rec

    # ── Итог ──
    show_session_summary(stats, student_id)
    client.close()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[dim]Выход[/dim]")
