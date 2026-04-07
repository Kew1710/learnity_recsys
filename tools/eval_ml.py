"""
Оценка качества ML-части рекомендательной системы Learnity.

Метрики:
  1. zpd_success_rate по кластерам — доля заданий, где ученик набрал score >= 0.7
  2. Средняя Δmastery за сессию (час) по кластерам — рост знаний
  3. Топ-KC по avg_reward в каждом кластере (из cluster_task_stats)
  4. Сравнение LinUCB vs эвристика (если есть данные A/B)

Запуск:
    python -m tools.eval_ml
    python -m tools.eval_ml --cluster 3   # подробно по одному кластеру
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from sqlalchemy import text

console = Console()


# ─── Запросы к БД ─────────────────────────────────────────────────────────────

async def fetch_zpd_success_by_cluster(session) -> list[dict]:
    """
    ZPD success rate: доля взаимодействий (score >= 0.7) по кластерам.
    Источник рекомендации — из bandit_log через kc_id и student_id.
    """
    rows = await session.execute(text("""
        SELECT
            sc.cluster_id,
            COUNT(*)                                             AS total,
            SUM(CASE WHEN i.score >= 0.7 THEN 1 ELSE 0 END)    AS success,
            AVG(i.score)                                         AS avg_score
        FROM interactions i
        JOIN student_clusters sc ON sc.student_id = i.student_id
        GROUP BY sc.cluster_id
        ORDER BY sc.cluster_id
    """))
    return [dict(r._mapping) for r in rows]


async def fetch_session_delta_by_cluster(session) -> list[dict]:
    """
    Средняя Δmastery за сессию (час) по кластерам.
    Берём из bandit_log где reward IS NOT NULL.
    """
    rows = await session.execute(text("""
        SELECT
            sc.cluster_id,
            COUNT(DISTINCT (bl.student_id, DATE_TRUNC('hour', bl.recommended_at))) AS sessions,
            AVG(session_total)                                   AS avg_session_delta,
            STDDEV(session_total)                                AS std_session_delta
        FROM (
            SELECT
                student_id,
                DATE_TRUNC('hour', recommended_at)  AS session_hour,
                SUM(reward)                          AS session_total
            FROM bandit_log
            WHERE reward IS NOT NULL
            GROUP BY student_id, DATE_TRUNC('hour', recommended_at)
        ) sessions
        JOIN student_clusters sc ON sc.student_id = sessions.student_id
        GROUP BY sc.cluster_id
        ORDER BY sc.cluster_id
    """))
    return [dict(r._mapping) for r in rows]


async def fetch_top_kcs_per_cluster(session, top_n: int = 5) -> dict[int, list[dict]]:
    """
    Топ-N KC по avg_reward для каждого кластера (из cluster_task_stats + bandit_log).
    """
    rows = await session.execute(text("""
        SELECT
            bl.kc_id,
            sc.cluster_id,
            AVG(bl.reward)  AS avg_reward,
            COUNT(*)        AS interactions
        FROM bandit_log bl
        JOIN student_clusters sc ON sc.student_id = bl.student_id
        WHERE bl.reward IS NOT NULL
        GROUP BY bl.kc_id, sc.cluster_id
        ORDER BY sc.cluster_id, AVG(bl.reward) DESC
    """))

    by_cluster: dict[int, list[dict]] = {}
    for r in rows:
        cid = int(r.cluster_id)
        if cid not in by_cluster:
            by_cluster[cid] = []
        if len(by_cluster[cid]) < top_n:
            by_cluster[cid].append({
                "kc_id": r.kc_id,
                "avg_reward": float(r.avg_reward or 0),
                "interactions": int(r.interactions),
            })
    return by_cluster


async def fetch_cluster_sizes(session) -> dict[int, int]:
    rows = await session.execute(text("""
        SELECT cluster_id, COUNT(*) AS cnt
        FROM student_clusters
        GROUP BY cluster_id
        ORDER BY cluster_id
    """))
    return {int(r.cluster_id): int(r.cnt) for r in rows}


async def fetch_linucb_learning_curve(session) -> list[dict]:
    """
    Динамика avg_reward по времени (объединяем все treatment-данные).
    Показывает, учится ли модель — растёт ли средний reward с накоплением данных.
    """
    rows = await session.execute(text("""
        SELECT
            DATE_TRUNC('day', bl.recommended_at)  AS day,
            COUNT(*)                               AS interactions,
            AVG(bl.reward)                         AS avg_reward
        FROM bandit_log bl
        JOIN student_experiments se ON se.student_id = bl.student_id
        WHERE bl.reward IS NOT NULL
          AND se.variant = 'treatment'
        GROUP BY DATE_TRUNC('day', bl.recommended_at)
        ORDER BY day
    """))
    return [dict(r._mapping) for r in rows]


async def fetch_model_coverage(session) -> dict:
    """Сколько KC имеют обученную LinUCB-модель vs сколько KC в системе."""
    trained = (await session.execute(
        text("SELECT COUNT(*) FROM bandit_model")
    )).scalar() or 0
    total_kc = (await session.execute(
        text("SELECT COUNT(DISTINCT kc_id) FROM mastery")
    )).scalar() or 0
    total_interactions = (await session.execute(
        text("SELECT COUNT(*) FROM bandit_log WHERE reward IS NOT NULL")
    )).scalar() or 0
    return {
        "trained_models": int(trained),
        "total_kc": int(total_kc),
        "total_interactions_with_reward": int(total_interactions),
    }


# ─── Отображение ──────────────────────────────────────────────────────────────

def show_zpd_success(rows: list[dict], cluster_sizes: dict[int, int]):
    console.print()
    table = Table(
        title="ZPD Success Rate по кластерам",
        box=box.ROUNDED,
        caption="score >= 0.7 считается успешным",
    )
    table.add_column("Кластер", justify="center")
    table.add_column("Студентов", justify="right")
    table.add_column("Взаимодействий", justify="right")
    table.add_column("Успешных", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Ср. скор", justify="right")

    if not rows:
        console.print("[dim]Нет данных о кластерах. Запусти python -m tools.cron_cluster[/dim]")
        return

    for r in rows:
        cid = int(r["cluster_id"])
        total = int(r["total"])
        success = int(r["success"])
        rate = success / total if total else 0
        avg = float(r["avg_score"] or 0)
        color = "green" if rate >= 0.65 else ("yellow" if rate >= 0.45 else "red")
        table.add_row(
            str(cid),
            str(cluster_sizes.get(cid, "—")),
            str(total),
            str(success),
            f"[{color}]{rate*100:.1f}%[/{color}]",
            f"{avg:.3f}",
        )

    console.print(table)


def show_session_deltas(rows: list[dict]):
    console.print()
    table = Table(
        title="Средняя Δmastery за сессию по кластерам",
        box=box.ROUNDED,
        caption="Сессия = все взаимодействия ученика за 1 час",
    )
    table.add_column("Кластер", justify="center")
    table.add_column("Сессий", justify="right")
    table.add_column("Ср. Δmastery", justify="right")
    table.add_column("Стандартное отклонение", justify="right")
    table.add_column("Оценка", justify="center")

    if not rows:
        console.print("[dim]Нет данных о сессиях в bandit_log[/dim]")
        return

    for r in rows:
        avg = float(r["avg_session_delta"] or 0)
        std = float(r["std_session_delta"] or 0)
        sessions = int(r["sessions"])
        color = "green" if avg >= 0.05 else ("yellow" if avg >= 0.01 else "red")
        grade = "Хорошо" if avg >= 0.05 else ("Средне" if avg >= 0.01 else "Низко")
        table.add_row(
            str(int(r["cluster_id"])),
            str(sessions),
            f"[{color}]{avg:+.4f}[/{color}]",
            f"{std:.4f}",
            f"[{color}]{grade}[/{color}]",
        )

    console.print(table)


def show_top_kcs(top_kcs: dict[int, list[dict]], focus_cluster: int | None = None):
    console.print()

    clusters_to_show = [focus_cluster] if focus_cluster is not None else sorted(top_kcs.keys())[:5]

    for cid in clusters_to_show:
        kcs = top_kcs.get(cid, [])
        if not kcs:
            continue

        table = Table(
            title=f"Топ KC — Кластер {cid}",
            box=box.SIMPLE_HEAD,
        )
        table.add_column("KC", style="dim")
        table.add_column("Ср. Δmastery", justify="right")
        table.add_column("Взаимодействий", justify="right")

        for kc in kcs:
            avg = kc["avg_reward"]
            color = "green" if avg >= 0.05 else ("yellow" if avg >= 0 else "red")
            table.add_row(
                kc["kc_id"],
                f"[{color}]{avg:+.4f}[/{color}]",
                str(kc["interactions"]),
            )

        console.print(table)


def show_learning_curve(rows: list[dict]):
    console.print()
    console.rule("[bold]Кривая обучения LinUCB (treatment-группа)[/bold]")
    console.print()

    if not rows:
        console.print("[dim]Нет данных treatment-группы. Нужно накопить взаимодействия.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_header=True)
    table.add_column("День")
    table.add_column("Взаимодействий", justify="right")
    table.add_column("Ср. награда", justify="right")
    table.add_column("Тренд", no_wrap=True)

    prev_avg = None
    for r in rows:
        avg = float(r["avg_reward"] or 0)
        cnt = int(r["interactions"])
        day = str(r["day"])[:10]
        if prev_avg is None:
            trend = "—"
        elif avg > prev_avg + 0.001:
            trend = f"[green]↑ +{avg - prev_avg:+.4f}[/green]"
        elif avg < prev_avg - 0.001:
            trend = f"[red]↓ {avg - prev_avg:+.4f}[/red]"
        else:
            trend = "[dim]→ стабильно[/dim]"
        color = "green" if avg >= 0.05 else ("yellow" if avg >= 0.01 else "dim")
        table.add_row(day, str(cnt), f"[{color}]{avg:+.4f}[/{color}]", trend)
        prev_avg = avg

    console.print(table)


def show_model_coverage(coverage: dict):
    console.print()
    trained = coverage["trained_models"]
    total = coverage["total_kc"]
    interactions = coverage["total_interactions_with_reward"]
    pct = trained / total * 100 if total else 0

    color = "green" if pct >= 50 else ("yellow" if pct >= 20 else "red")

    console.print(Panel(
        f"Обученных LinUCB-моделей: [{color}]{trained}[/{color}] / {total} KC ({pct:.1f}%)\n"
        f"Взаимодействий с наградой: [cyan]{interactions}[/cyan]",
        title="Покрытие моделей",
        border_style="cyan",
        padding=(0, 2),
    ))


# ─── Главная точка входа ──────────────────────────────────────────────────────

async def main(focus_cluster: int | None):
    from shared.db import AsyncSessionLocal

    console.print()
    console.print(Panel(
        "[bold white]Learnity — ML Evaluation[/bold white]\n"
        "[dim]ZPD success rate, Δmastery по кластерам, кривая обучения LinUCB[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

    try:
        async with AsyncSessionLocal() as session:
            coverage = await fetch_model_coverage(session)
            show_model_coverage(coverage)

            cluster_sizes = await fetch_cluster_sizes(session)
            zpd_rows = await fetch_zpd_success_by_cluster(session)
            show_zpd_success(zpd_rows, cluster_sizes)

            delta_rows = await fetch_session_delta_by_cluster(session)
            show_session_deltas(delta_rows)

            top_kcs = await fetch_top_kcs_per_cluster(session)
            show_top_kcs(top_kcs, focus_cluster)

            curve = await fetch_linucb_learning_curve(session)
            show_learning_curve(curve)

    except Exception as e:
        console.print(f"\n[red]Ошибка подключения к БД: {e}[/red]")
        console.print("[dim]Убедись что PostgreSQL запущен: make up[/dim]")
        sys.exit(1)

    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML eval для Learnity")
    parser.add_argument("--cluster", type=int, default=None, help="Показать детали по одному кластеру")
    args = parser.parse_args()
    asyncio.run(main(args.cluster))
