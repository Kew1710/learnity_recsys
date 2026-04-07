#!/usr/bin/env python3
"""
Полностью автоматический A/B тест: LinUCB vs Baseline.

Запуск:
    python tools/run_ab.py                         # дефолты: seed=42, n=30, tasks=250
    python tools/run_ab.py --seed 7 --n 10 --tasks 100
    python tools/run_ab.py --skip-warmup
    python tools/run_ab.py --n 5 --tasks 50 --warmup-n 5 --warmup-tasks 30  # быстрый тест

Требования:
    - Все сервисы кроме retrieval должны быть запущены (make dev, потом Ctrl-C retrieval не нужен)
    - Скрипт сам управляет retrieval: убивает, перезапускает с нужным ENABLE_CONTROL_GROUP
"""

from __future__ import annotations

import argparse
from typing import Optional
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.panel import Panel
import random

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.simulation import (
    GRADE_POOL, PRESET_PROFILES,
    run_population, run_warmup, save_session_summary,
    show_ab_comparison, show_population_results, show_cluster_alignment,
    _sample_profile_ab,
    console,
)

RETRIEVAL_URL = "http://localhost:8004"
GATEWAY_URL   = "http://localhost:8005"
MACRO_URL     = "http://localhost:8006"
UVICORN       = str(ROOT / ".venv" / "bin" / "uvicorn")
PID_FILE      = ROOT / ".pids" / "retrieval.pid"
LOG_FILE      = ROOT / "logs" / "retrieval.log"


# ─── Управление retrieval-сервисом ───────────────────────────────────────────

def _kill_retrieval() -> None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except (ProcessLookupError, ValueError, OSError):
            pass
        PID_FILE.unlink(missing_ok=True)


def _start_retrieval(control_group: bool) -> None:
    env = os.environ.copy()
    env.update({
        "PROFILE_URL":           "http://localhost:8001",
        "GRAPH_URL":             "http://localhost:8002",
        "TASK_BANK_URL":         "http://localhost:8003",
        "ENABLE_CONTROL_GROUP":  "true" if control_group else "false",
    })
    log_f = LOG_FILE.open("w")
    proc = subprocess.Popen(
        [UVICORN, "services.retrieval.main:app", "--port", "8004"],
        env=env,
        stdout=log_f,
        stderr=log_f,
        cwd=str(ROOT),
    )
    PID_FILE.write_text(str(proc.pid))

    label = "Baseline" if control_group else "LinUCB"
    console.print(f"  [dim]Жду запуска retrieval ({label})[/dim]", end="")
    for _ in range(30):
        time.sleep(1)
        try:
            r = httpx.get(f"{RETRIEVAL_URL}/health", timeout=2, trust_env=False)
            if r.status_code == 200:
                console.print("  [green]✓[/green]")
                return
        except Exception:
            pass
        console.print(".", end="")
    console.print()
    raise RuntimeError(f"Retrieval ({label}) не запустился за 30 секунд — проверь logs/retrieval.log")


def _restart_retrieval(control_group: bool) -> None:
    label = "Baseline  (ENABLE_CONTROL_GROUP=true)" if control_group else "LinUCB   (ENABLE_CONTROL_GROUP=false)"
    console.print(f"\n[bold cyan]↺ Retrieval → {label}[/bold cyan]")
    _kill_retrieval()
    _start_retrieval(control_group)


def _preflight() -> None:
    """Проверяет что все зависимые сервисы запущены. Завершает процесс с подсказкой если нет."""
    required = {
        "Gateway   :8005": "http://localhost:8005/health",
        "Profile   :8001": "http://localhost:8001/health",
        "Graph     :8002": "http://localhost:8002/health",
        "Task Bank :8003": "http://localhost:8003/health",
        "Macro     :8006": "http://localhost:8006/health",
    }
    # Отключаем proxy — сервисы локальные, proxy даст 502
    no_proxy_client = httpx.Client(trust_env=False, timeout=5)
    missing = []
    for name, url in required.items():
        try:
            r = no_proxy_client.get(url)
            if r.status_code == 200:
                console.print(f"  [green]✓[/green] [dim]{name}[/dim]")
            else:
                missing.append(name)
                console.print(f"  [red]✗[/red] [dim]{name}[/dim] (HTTP {r.status_code})")
        except Exception:
            missing.append(name)
            console.print(f"  [red]✗[/red] [dim]{name}[/dim] (не отвечает)")
    no_proxy_client.close()

    if missing:
        console.print(
            f"\n[red bold]Запусти сервисы перед A/B тестом:[/red bold]\n\n"
            "  [bold]make up && make dev[/bold]   # make up — Neo4j/Postgres в Docker\n\n"
            "Retrieval скрипт запустит сам — его можно не включать в make dev.\n"
        )
        sys.exit(1)


# ─── Сборка популяции студентов ───────────────────────────────────────────────

def _build_profiles(n_per_group: int, rng_seed: int) -> list:
    """Одинаковый seed → одинаковые профили в обоих прогонах."""
    rng = random.Random(rng_seed)
    profiles = []
    for base in PRESET_PROFILES.values():
        profiles.extend(_sample_profile_ab(base, i, rng) for i in range(n_per_group))
    return profiles


# ─── Основной пайплайн ────────────────────────────────────────────────────────

def run_ab(
    seed: int = 42,
    n_per_group: int = 30,
    max_tasks: int = 250,
    warmup: bool = True,
    warmup_n: int = 20,
    warmup_tasks: int = 60,
    variant: str = "both",  # "both" | "linucb" | "baseline"
    plan_config: Optional[dict] = None,
) -> None:
    started_at = datetime.now(timezone.utc)

    console.print("\n[dim]Проверка сервисов...[/dim]")
    _preflight()

    label_map = {"both": "A/B тест: LinUCB vs Baseline", "linucb": "Прогон: LinUCB", "baseline": "Прогон: Baseline"}
    console.print()
    console.print(Panel(
        f"[bold]{label_map[variant]}[/bold]\n\n"
        f"  seed={seed}  n_per_group={n_per_group}  max_tasks={max_tasks}\n"
        f"  warmup: {'да' if warmup else 'нет'}"
        + (f" ({warmup_n}×4 группы × {warmup_tasks} заданий)" if warmup else ""),
        border_style="magenta", padding=(0, 2),
    ))

    client = httpx.Client(base_url=GATEWAY_URL, timeout=60, trust_env=False)

    # ── 1. Прогрев ────────────────────────────────────────────────────────────
    if warmup:
        console.print("\n[bold]── Фаза 0: Прогрев кластеров ──[/bold]")
        _restart_retrieval(control_group=False)
        run_warmup(client, n_per_group=warmup_n, max_tasks=warmup_tasks)

    saved_paths = []

    # ── 2. Прогон LinUCB ─────────────────────────────────────────────────────
    if variant in ("both", "linucb"):
        _restart_retrieval(control_group=False)
        ts_a = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag_a = f"ab_linucb_{ts_a}"
        profiles_a = _build_profiles(n_per_group, seed)

        console.print(f"\n[bold]── Прогон LinUCB  ({len(profiles_a)} студентов × {max_tasks} заданий) ──[/bold]\n")
        results_a = run_population(client, profiles_a, max_tasks, session_tag=tag_a, plan_config=plan_config)
        show_population_results(results_a, group_name="LinUCB", grouped=True)
        show_cluster_alignment(results_a)
        saved_paths.append(save_session_summary(results_a, "linucb", tag_a, seed, max_tasks))

    # ── 3. Прогон Baseline ───────────────────────────────────────────────────
    if variant in ("both", "baseline"):
        _restart_retrieval(control_group=True)
        ts_b = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag_b = f"ab_baseline_{ts_b}"
        profiles_b = _build_profiles(n_per_group, seed)

        console.print(f"\n[bold]── Прогон Baseline  ({len(profiles_b)} студентов × {max_tasks} заданий) ──[/bold]\n")
        results_b = run_population(client, profiles_b, max_tasks, session_tag=tag_b, plan_config=plan_config)
        show_population_results(results_b, group_name="Baseline", grouped=True)
        show_cluster_alignment(results_b)
        saved_paths.append(save_session_summary(results_b, "baseline", tag_b, seed, max_tasks))

    # ── 4. Сравнение (только если оба прогона) ───────────────────────────────
    if variant == "both":
        console.print("\n[bold]── Итоговое сравнение ──[/bold]")
        show_ab_comparison()

    # ── 5. Восстановить retrieval в обычный режим ────────────────────────────
    _restart_retrieval(control_group=False)

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    paths_str = "\n".join(f"  [dim]{p}[/dim]" for p in saved_paths)
    console.print(f"\n[green bold]✓ Завершено за {elapsed/60:.1f} мин[/green bold]\n{paths_str}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B тест: LinUCB vs Baseline — полностью автоматический",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed",         type=int, default=42,  help="RNG seed (одинаковый для обоих прогонов)")
    parser.add_argument("--n",            type=int, default=30,  help="Студентов из каждой группы")
    parser.add_argument("--tasks",        type=int, default=250, help="Заданий на студента")
    parser.add_argument("--skip-warmup",  action="store_true",   help="Пропустить прогрев кластеров")
    parser.add_argument("--warmup-n",     type=int, default=20,  help="Студентов на группу в прогреве")
    parser.add_argument("--warmup-tasks", type=int, default=60,  help="Заданий в прогреве")
    parser.add_argument("--variant",      choices=["both", "linucb", "baseline"], default="both",
                        help="Какой вариант запустить: both=A/B, linucb=только LinUCB, baseline=только Baseline")
    parser.add_argument("--plan-mode",    type=str, default=None,
                        choices=["target_mastery", "coverage"],
                        help="Режим плана: target_mastery | coverage")
    parser.add_argument("--plan-kc",      type=str, default=None,
                        help="Целевая KC для режима target_mastery (например kc_quadratic_eq)")
    parser.add_argument("--plan-variant", type=str, default=None,
                        choices=["count", "mass", "frontier"],
                        help="Вариант coverage: count | mass | frontier")
    parser.add_argument("--plan-budget",  type=int, default=None,
                        help="Бюджет заданий для режима coverage")
    parser.add_argument("--plan-mastery-threshold", type=float, default=None,
                        help="Порог освоения KC в плане (default=0.80). 0.85 включает KC grade-1 в план")
    args = parser.parse_args()

    plan_config: Optional[dict] = None
    if args.plan_mode:
        plan_config = {"mode": args.plan_mode}
        if args.plan_kc:
            plan_config["target_kc_id"] = args.plan_kc
        if args.plan_variant:
            plan_config["coverage_variant"] = args.plan_variant
        if args.plan_budget:
            plan_config["task_budget"] = args.plan_budget
        if args.plan_mastery_threshold:
            plan_config["mastery_threshold"] = args.plan_mastery_threshold

    run_ab(
        seed=args.seed,
        n_per_group=args.n,
        max_tasks=args.tasks,
        warmup=not args.skip_warmup,
        warmup_n=args.warmup_n,
        warmup_tasks=args.warmup_tasks,
        variant=args.variant,
        plan_config=plan_config,
    )
