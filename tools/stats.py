#!/usr/bin/env python3
"""
Аналитический дашборд рекомендательной системы.

Запуск:
    python tools/stats.py
    make stats

Показывает:
    1. Обзор студентов
    2. Mastery топ/антитоп по KC
    3. Гипотезы: ZPD vs Exploration, subject distribution
    4. Детальная история взаимодействий (по студенту)
"""

import asyncio
import sys
from pathlib import Path
from collections import defaultdict

# Добавляем корень проекта в sys.path (tools/ запускается не из корня)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from sqlalchemy import text

console = Console()

# KC-метаданные из граф-сида — для читаемых имён и предметов
try:
    from services.graph.seed import KCS
    KC_NAME = {kc["kc_id"]: kc["name"] for kc in KCS}
    KC_SUBJECT = {kc["kc_id"]: kc["subject"] for kc in KCS}
except ImportError:
    KC_NAME = {}
    KC_SUBJECT = {}

SUBJECT_COLORS = {
    "arithmetic": "cyan",
    "algebra": "green",
    "geometry": "yellow",
    "statistics": "magenta",
}
SUBJECT_RU = {
    "arithmetic": "Арифметика",
    "algebra": "Алгебра",
    "geometry": "Геометрия",
    "statistics": "Статистика",
}


def mastery_bar(value: float, width: int = 8) -> str:
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def pct_color(value: float) -> str:
    if value >= 0.7:
        return "green"
    if value >= 0.4:
        return "yellow"
    return "red"


# ─── Запросы к БД ─────────────────────────────────────────────────────────────

async def fetch_students(session) -> list[dict]:
    rows = await session.execute(text("""
        SELECT
            s.student_id,
            s.grade,
            s.review_mode,
            s.created_at,
            COUNT(DISTINCT m.kc_id)    AS kc_count,
            AVG(m.probability)         AS avg_mastery,
            COUNT(DISTINCT i.task_id)          AS tasks_done,
            COUNT(i.interaction_id)            AS interactions_count
        FROM students s
        LEFT JOIN mastery  m ON m.student_id = s.student_id
        LEFT JOIN interactions i ON i.student_id = s.student_id
        GROUP BY s.student_id, s.grade, s.review_mode, s.created_at
        ORDER BY s.created_at DESC
    """))
    return [dict(r._mapping) for r in rows]


async def fetch_mastery_for_student(session, student_id: str) -> list[dict]:
    rows = await session.execute(text("""
        SELECT kc_id, probability, last_practiced, attempts_count
        FROM mastery
        WHERE student_id = :sid
        ORDER BY probability DESC
    """), {"sid": student_id})
    return [dict(r._mapping) for r in rows]


async def fetch_interactions(session, student_id: str, limit: int = 30) -> list[dict]:
    rows = await session.execute(text("""
        SELECT task_id, part_id, score, recommendation_source, timestamp, misconception_triggered
        FROM interactions
        WHERE student_id = :sid
        ORDER BY timestamp DESC
        LIMIT :lim
    """), {"sid": student_id, "lim": limit})
    return [dict(r._mapping) for r in rows]


async def fetch_kc_stats(session) -> list[dict]:
    """Агрегация по KC из таблицы mastery (средняя освоенность)."""
    rows = await session.execute(text("""
        SELECT kc_id,
               AVG(probability)         AS avg_mastery,
               AVG(attempts_count)      AS avg_attempts,
               COUNT(DISTINCT student_id) AS students_count
        FROM mastery
        GROUP BY kc_id
        HAVING COUNT(DISTINCT student_id) > 0
        ORDER BY avg_mastery DESC
    """))
    return [dict(r._mapping) for r in rows]


async def fetch_source_distribution(session) -> dict:
    """ZPD vs exploration по всем взаимодействиям."""
    rows = await session.execute(text("""
        SELECT
            recommendation_source,
            COUNT(*)         AS cnt,
            AVG(score)       AS avg_score,
            SUM(CASE WHEN score >= 0.7 THEN 1 ELSE 0 END) AS success_count
        FROM interactions
        WHERE recommendation_source IS NOT NULL
        GROUP BY recommendation_source
    """))
    result = {}
    for r in rows:
        result[r.recommendation_source] = {
            "count": r.cnt,
            "avg_score": float(r.avg_score or 0),
            "success_count": r.success_count,
        }
    return result


async def fetch_subject_distribution(session) -> dict:
    """Распределение взаимодействий по предметам (через масtery kc_id)."""
    rows = await session.execute(text("""
        SELECT part_id, COUNT(*) as cnt, AVG(score) as avg_score
        FROM interactions
        GROUP BY part_id
    """))
    # part_id = "p1" для всех стабов, не несёт информации о предмете
    # Поэтому используем mastery для subject distribution
    rows2 = await session.execute(text("""
        SELECT kc_id, SUM(attempts_count) as total_attempts
        FROM mastery
        GROUP BY kc_id
    """))
    subject_attempts: dict[str, int] = defaultdict(int)
    for r in rows2:
        subject = KC_SUBJECT.get(r.kc_id, "unknown")
        subject_attempts[subject] += int(r.total_attempts or 0)
    return dict(subject_attempts)


async def fetch_mastery_progress(session, student_id: str) -> list[dict]:
    """История mastery во времени через interactions (по timestamp)."""
    rows = await session.execute(text("""
        SELECT
            DATE_TRUNC('hour', timestamp) AS hour,
            COUNT(*) AS count,
            AVG(score) AS avg_score
        FROM interactions
        WHERE student_id = :sid
        GROUP BY DATE_TRUNC('hour', timestamp)
        ORDER BY hour
    """), {"sid": student_id})
    return [dict(r._mapping) for r in rows]


async def fetch_operational_metrics(session) -> dict:
    metrics: dict[str, float | int] = {}

    bandit_row = (await session.execute(text("""
        SELECT
            COUNT(*) AS total_recommendations,
            SUM(CASE WHEN fallback_occurred THEN 1 ELSE 0 END) AS plan_fallbacks,
            SUM(CASE WHEN irt_fallback_occurred THEN 1 ELSE 0 END) AS irt_fallbacks
        FROM bandit_log
    """))).one()
    metrics["total_recommendations"] = int(bandit_row.total_recommendations or 0)
    metrics["plan_fallbacks"] = int(bandit_row.plan_fallbacks or 0)
    metrics["irt_fallbacks"] = int(bandit_row.irt_fallbacks or 0)

    alert_row = (await session.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE alert_type = 'content_gap') AS content_gap_alerts,
            COUNT(*) FILTER (WHERE alert_type = 'plateau') AS plateau_alerts,
            COUNT(*) FILTER (WHERE alert_type = 'replan_requested') AS replan_alerts
        FROM teacher_alerts
    """))).one()
    metrics["content_gap_alerts"] = int(alert_row.content_gap_alerts or 0)
    metrics["plateau_alerts"] = int(alert_row.plateau_alerts or 0)
    metrics["replan_alerts"] = int(alert_row.replan_alerts or 0)

    cat_row = (await session.execute(text("""
        SELECT
            COUNT(*) AS total_cat_sessions,
            SUM(CASE WHEN completed THEN 1 ELSE 0 END) AS completed_cat_sessions,
            AVG(tasks_used) FILTER (WHERE completed) AS avg_cat_tasks
        FROM diagnostic_cat_state
    """))).one()
    metrics["total_cat_sessions"] = int(cat_row.total_cat_sessions or 0)
    metrics["completed_cat_sessions"] = int(cat_row.completed_cat_sessions or 0)
    metrics["avg_cat_tasks"] = float(cat_row.avg_cat_tasks or 0.0)

    cluster_row = (await session.execute(text("""
        SELECT
            COUNT(*) AS total_cluster_events,
            COUNT(*) FILTER (WHERE from_cluster_id IS NOT NULL AND from_cluster_id != to_cluster_id) AS cluster_shifts
        FROM student_cluster_history
    """))).one()
    metrics["total_cluster_events"] = int(cluster_row.total_cluster_events or 0)
    metrics["cluster_shifts"] = int(cluster_row.cluster_shifts or 0)

    outcome_row = (await session.execute(text("""
        SELECT
            COUNT(*) AS total_step_outcomes,
            COUNT(*) FILTER (WHERE outcome_type = 'completed') AS completed_steps,
            COUNT(*) FILTER (WHERE outcome_type = 'replan_requested') AS replan_step_events,
            COUNT(*) FILTER (WHERE outcome_type = 'plateau') AS plateau_step_events,
            COUNT(*) FILTER (WHERE outcome_type = 'regression_reopen') AS regression_reopens,
            AVG(tasks_spent) FILTER (WHERE outcome_type = 'completed') AS avg_tasks_to_complete
        FROM macro_step_outcomes
    """))).one()
    metrics["total_step_outcomes"] = int(outcome_row.total_step_outcomes or 0)
    metrics["completed_steps"] = int(outcome_row.completed_steps or 0)
    metrics["replan_step_events"] = int(outcome_row.replan_step_events or 0)
    metrics["plateau_step_events"] = int(outcome_row.plateau_step_events or 0)
    metrics["regression_reopens"] = int(outcome_row.regression_reopens or 0)
    metrics["avg_tasks_to_complete"] = float(outcome_row.avg_tasks_to_complete or 0.0)

    return metrics


# ─── Отображение ──────────────────────────────────────────────────────────────

def show_students_table(students: list[dict]):
    if not students:
        console.print("[dim]Нет студентов в базе[/dim]")
        return

    table = Table(
        title=f"👥 Студенты ({len(students)})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("student_id", style="dim", max_width=36)
    table.add_column("Класс", justify="center")
    table.add_column("KC освоено", justify="right")
    table.add_column("Ср. mastery", justify="right")
    table.add_column("Заданий", justify="right")
    table.add_column("Взаимод.", justify="right")
    table.add_column("review", justify="center")
    table.add_column("Прогресс", no_wrap=True)

    for s in students:
        avg = float(s["avg_mastery"] or 0.0)
        color = pct_color(avg)
        table.add_row(
            str(s["student_id"]),
            str(s["grade"]),
            str(s["kc_count"] or 0),
            f"[{color}]{avg:.3f}[/{color}]",
            str(s["tasks_done"] or 0),
            str(s["interactions_count"] or 0),
            "✓" if s["review_mode"] else "—",
            mastery_bar(avg),
        )

    console.print()
    console.print(table)


def show_kc_stats(kc_stats: list[dict]):
    if not kc_stats:
        console.print("[dim]Нет данных по KC[/dim]")
        return

    # Топ-10 и антитоп-10
    top = kc_stats[:10]
    bottom = kc_stats[-10:][::-1]  # наименее освоенные

    for label, rows, title_color in [
        ("🏆 Топ-10 освоенных KC", top, "green"),
        ("⚠️  Антитоп-10 (наименее освоенные)", bottom, "red"),
    ]:
        table = Table(title=label, box=box.SIMPLE_HEAD, show_header=True)
        table.add_column("KC", style="dim", max_width=38)
        table.add_column("Название", max_width=40)
        table.add_column("Ср. mastery", justify="right")
        table.add_column("Попыток (avg)", justify="right")
        table.add_column("Студентов", justify="right")
        table.add_column("Прогресс", no_wrap=True)

        for r in rows:
            avg = float(r["avg_mastery"] or 0.0)
            color = pct_color(avg)
            subj = KC_SUBJECT.get(r["kc_id"], "")
            subj_color = SUBJECT_COLORS.get(subj, "white")
            name = KC_NAME.get(r["kc_id"], "—")
            table.add_row(
                f"[{subj_color}]{r['kc_id']}[/{subj_color}]",
                name[:40],
                f"[{color}]{avg:.3f}[/{color}]",
                f"{float(r['avg_attempts'] or 0):.1f}",
                str(r["students_count"]),
                mastery_bar(avg),
            )

        console.print()
        console.print(table)


def show_hypotheses(source_dist: dict, subject_dist: dict):
    console.print()
    console.rule("[bold]📊 Гипотезы и метрики[/bold]")

    # ── Гипотеза 1: ZPD vs Exploration ──
    zpd = source_dist.get("zpd", {})
    exp = source_dist.get("exploration", {})

    console.print()
    console.print("[bold]Гипотеза 1:[/bold] ZPD (exploitation) даёт лучший скор чем Exploration")
    console.print()

    h1_table = Table(box=box.SIMPLE, show_header=True)
    h1_table.add_column("Источник", style="bold")
    h1_table.add_column("Заданий", justify="right")
    h1_table.add_column("Ср. скор", justify="right")
    h1_table.add_column("Успех (≥0.7)", justify="right")
    h1_table.add_column("Успех %", justify="right")

    for label, data, color in [
        ("ZPD",        zpd, "green"),
        ("Exploration", exp, "yellow"),
    ]:
        cnt = data.get("count", 0)
        avg = data.get("avg_score", 0.0)
        succ = data.get("success_count", 0)
        succ_pct = succ / cnt * 100 if cnt > 0 else 0.0
        h1_table.add_row(
            f"[{color}]{label}[/{color}]",
            str(cnt),
            f"{avg:.3f}",
            str(succ),
            f"{succ_pct:.1f}%",
        )

    console.print(h1_table)

    if zpd and exp:
        zpd_score = zpd.get("avg_score", 0.0)
        exp_score = exp.get("avg_score", 0.0)
        if zpd_score > exp_score + 0.05:
            console.print("  [green]→ ZPD показывает лучший скор. Гипотеза подтверждается.[/green]")
        elif abs(zpd_score - exp_score) <= 0.05:
            console.print("  [yellow]→ Разница незначительна (<0.05). Нужно больше данных.[/yellow]")
        else:
            console.print("  [red]→ Exploration даёт лучший скор. Возможно, ZPD ставит слишком сложные задачи.[/red]")
    else:
        console.print("  [dim]Недостаточно данных для сравнения[/dim]")

    # ── Гипотеза 2: Subject rotation ──
    console.print()
    console.print("[bold]Гипотеза 2:[/bold] Subject rotation обеспечивает равномерный охват предметов")
    console.print()

    if subject_dist:
        total = sum(subject_dist.values())
        h2_table = Table(box=box.SIMPLE, show_header=True)
        h2_table.add_column("Предмет")
        h2_table.add_column("Попыток", justify="right")
        h2_table.add_column("Доля", justify="right")
        h2_table.add_column("Распределение", no_wrap=True)

        for subject, count in sorted(subject_dist.items(), key=lambda x: -x[1]):
            if subject == "unknown":
                continue
            color = SUBJECT_COLORS.get(subject, "white")
            ru = SUBJECT_RU.get(subject, subject)
            pct = count / total if total > 0 else 0
            bar_len = round(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            h2_table.add_row(
                f"[{color}]{ru}[/{color}]",
                str(count),
                f"{pct*100:.1f}%",
                bar,
            )

        console.print(h2_table)

        pcts = [v / total for v in subject_dist.values() if v > 0 and "unknown" not in str(v)]
        expected = 1.0 / len(pcts) if pcts else 0
        max_deviation = max(abs(p - expected) for p in pcts) if pcts else 0
        if max_deviation < 0.10:
            console.print("  [green]→ Равномерное распределение. Rotation работает.[/green]")
        elif max_deviation < 0.25:
            console.print("  [yellow]→ Небольшое смещение. Нормально на малом наборе данных.[/yellow]")
        else:
            console.print("  [red]→ Значительный перекос. Возможно, в ZPD доминирует один предмет.[/red]")
    else:
        console.print("  [dim]Нет данных о взаимодействиях[/dim]")


def show_interactions(interactions: list[dict], student_id: str):
    if not interactions:
        console.print("[dim]Нет взаимодействий[/dim]")
        return

    table = Table(
        title=f"История взаимодействий (последние {len(interactions)})",
        box=box.SIMPLE_HEAD,
        show_header=True,
    )
    table.add_column("Время", style="dim", max_width=16)
    table.add_column("task_id", style="dim", max_width=8)
    table.add_column("Скор", justify="right")
    table.add_column("Источник")
    table.add_column("Рез-т", justify="center")

    for i in interactions:
        score = float(i["score"])
        color = pct_color(score)
        result = "✓" if score >= 0.7 else ("△" if score >= 0.4 else "✗")
        ts = str(i["timestamp"])[:16]
        tid = str(i["task_id"])[:8] + "…"
        src = i["recommendation_source"] or "—"
        src_colored = f"[green]{src}[/green]" if src == "zpd" else f"[yellow]{src}[/yellow]"

        table.add_row(ts, tid, f"[{color}]{score:.2f}[/{color}]", src_colored, result)

    console.print()
    console.print(table)


def show_operational_metrics(metrics: dict):
    console.print()
    console.rule("[bold]🩺 Operational Signals[/bold]")

    total_recs = int(metrics.get("total_recommendations", 0))
    plan_fallbacks = int(metrics.get("plan_fallbacks", 0))
    irt_fallbacks = int(metrics.get("irt_fallbacks", 0))
    total_cat = int(metrics.get("total_cat_sessions", 0))
    completed_cat = int(metrics.get("completed_cat_sessions", 0))
    cluster_events = int(metrics.get("total_cluster_events", 0))
    cluster_shifts = int(metrics.get("cluster_shifts", 0))

    table = Table(box=box.SIMPLE_HEAD, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Rate", justify="right")

    def _pct(num: int, den: int) -> str:
        return f"{(num / den * 100):.1f}%" if den > 0 else "—"

    table.add_row("Recommendations", str(total_recs), "—")
    table.add_row("Plan fallback", str(plan_fallbacks), _pct(plan_fallbacks, total_recs))
    table.add_row("IRT fallback", str(irt_fallbacks), _pct(irt_fallbacks, total_recs))
    table.add_row("Content gap alerts", str(int(metrics.get("content_gap_alerts", 0))), "—")
    table.add_row("Plateau alerts", str(int(metrics.get("plateau_alerts", 0))), "—")
    table.add_row("Replan alerts", str(int(metrics.get("replan_alerts", 0))), "—")
    table.add_row("CAT completed", str(completed_cat), _pct(completed_cat, total_cat))
    table.add_row("Avg CAT tasks", f"{float(metrics.get('avg_cat_tasks', 0.0)):.2f}", "—")
    table.add_row("Cluster shifts", str(cluster_shifts), _pct(cluster_shifts, cluster_events))
    table.add_row("Macro step outcomes", str(int(metrics.get("total_step_outcomes", 0))), "—")
    table.add_row("Completed steps", str(int(metrics.get("completed_steps", 0))), "—")
    table.add_row("Avg tasks to complete", f"{float(metrics.get('avg_tasks_to_complete', 0.0)):.2f}", "—")
    table.add_row("Replan step events", str(int(metrics.get("replan_step_events", 0))), "—")
    table.add_row("Plateau step events", str(int(metrics.get("plateau_step_events", 0))), "—")
    table.add_row("Regression reopens", str(int(metrics.get("regression_reopens", 0))), "—")
    console.print(table)


# ─── Главная точка входа ──────────────────────────────────────────────────────

async def main():
    from shared.db import AsyncSessionLocal

    console.print()
    console.print(Panel(
        "[bold white]Learnity — Аналитический дашборд[/bold white]\n"
        "[dim]Метрики рекомендательной системы и проверка гипотез[/dim]",
        border_style="magenta",
        padding=(0, 2),
    ))

    try:
        async with AsyncSessionLocal() as session:
            students = await fetch_students(session)
            show_students_table(students)

            if not students:
                console.print("\n[dim]База пуста. Запусти make play чтобы создать студентов.[/dim]")
                return

            # KC stats
            kc_stats = await fetch_kc_stats(session)
            show_kc_stats(kc_stats)

            # Гипотезы
            source_dist = await fetch_source_distribution(session)
            subject_dist = await fetch_subject_distribution(session)
            show_hypotheses(source_dist, subject_dist)
            op_metrics = await fetch_operational_metrics(session)
            show_operational_metrics(op_metrics)

            # Детальная история для выбранного студента
            if len(students) == 1:
                chosen_id = str(students[0]["student_id"])
            else:
                console.print()
                console.print("[dim]Для детальной истории введи student_id (или Enter для первого):[/dim]")
                for i, s in enumerate(students[:5]):
                    console.print(f"  [{i}] {s['student_id']}  класс {s['grade']}  {s['interactions_count']} взаимод.")
                raw = console.input("> ").strip()
                if raw == "" or raw == "0":
                    chosen_id = str(students[0]["student_id"])
                elif raw.isdigit() and int(raw) < len(students):
                    chosen_id = str(students[int(raw)]["student_id"])
                else:
                    chosen_id = raw

            console.print(f"\n[dim]История: {chosen_id}[/dim]")
            interactions = await fetch_interactions(session, chosen_id)
            show_interactions(interactions, chosen_id)

            # Mastery для выбранного студента
            mastery = await fetch_mastery_for_student(session, chosen_id)
            if mastery:
                console.print()
                m_table = Table(
                    title=f"Mastery студента (топ-20)",
                    box=box.SIMPLE_HEAD, show_header=True,
                )
                m_table.add_column("KC")
                m_table.add_column("Название", max_width=38)
                m_table.add_column("P(знает)", justify="right")
                m_table.add_column("Попыток", justify="right")
                m_table.add_column("Прогресс")

                for r in mastery[:20]:
                    p = float(r["probability"])
                    color = pct_color(p)
                    subj = KC_SUBJECT.get(r["kc_id"], "")
                    subj_color = SUBJECT_COLORS.get(subj, "white")
                    name = KC_NAME.get(r["kc_id"], "—")
                    m_table.add_row(
                        f"[{subj_color}]{r['kc_id']}[/{subj_color}]",
                        name[:38],
                        f"[{color}]{p:.3f}[/{color}]",
                        str(r["attempts_count"]),
                        mastery_bar(p),
                    )

                if len(mastery) > 20:
                    m_table.add_row("...", f"(ещё {len(mastery) - 20})", "", "", "")

                console.print(m_table)

    except Exception as e:
        console.print(f"\n[red]Ошибка подключения к БД: {e}[/red]")
        console.print("[dim]Убедись что PostgreSQL запущен: make up[/dim]")
        sys.exit(1)

    console.print()


if __name__ == "__main__":
    asyncio.run(main())
