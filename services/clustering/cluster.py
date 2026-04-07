"""
Кластеризация учеников по mastery-векторам.

run_clustering() — запускается раз в час из cron:
  1. Читает все mastery-записи из БД
  2. Строит матрицу (студент × KC)
  3. Запускает KMeans(k=15)
  4. Записывает результаты в student_clusters
  5. Пересчитывает cluster_task_stats из bandit_log
  6. Сохраняет центроиды на диск для немедленного назначения новых студентов

assign_cluster_for_new_student(mastery) — вызывается при создании ученика:
  Возвращает cluster_id по ближайшему центроиду. None если центроиды ещё не вычислены.
"""

import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import sqlalchemy as sa
from sklearn.cluster import KMeans

from shared.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

N_CLUSTERS = 15
_CENTROIDS_PATH = os.getenv("CLUSTER_CENTROIDS_PATH", "/tmp/learnity_centroids.npy")
_KC_ORDER_PATH = os.getenv("CLUSTER_KC_ORDER_PATH", "/tmp/learnity_kc_order.npy")


async def run_clustering() -> None:
    """Полный цикл кластеризации. Идемпотентен — безопасно запускать повторно."""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("SELECT student_id, kc_id, probability FROM mastery")
        )).fetchall()

    if not rows:
        logger.info("clustering: no mastery records, skipping")
        return

    # Строим матрицу студент × KC
    student_ids = sorted({r[0] for r in rows})
    kc_ids = sorted({r[1] for r in rows})
    student_idx = {s: i for i, s in enumerate(student_ids)}
    kc_idx = {k: i for i, k in enumerate(kc_ids)}

    matrix = np.zeros((len(student_ids), len(kc_ids)), dtype=np.float32)
    for student_id, kc_id, prob in rows:
        matrix[student_idx[student_id], kc_idx[kc_id]] = float(prob)

    n_clusters = min(N_CLUSTERS, len(student_ids))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(matrix)

    # Сохраняем центроиды для немедленного назначения новых учеников
    np.save(_CENTROIDS_PATH, kmeans.cluster_centers_.astype(np.float32))
    np.save(_KC_ORDER_PATH, np.array(kc_ids, dtype=object))

    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        # Обновляем student_clusters
        for i, student_id in enumerate(student_ids):
            await db.execute(
                sa.text("""
                    INSERT INTO student_clusters (student_id, cluster_id, assigned_at)
                    VALUES (:student_id, :cluster_id, :assigned_at)
                    ON CONFLICT (student_id)
                    DO UPDATE SET cluster_id = EXCLUDED.cluster_id,
                                  assigned_at = EXCLUDED.assigned_at
                """),
                {
                    "student_id": student_id,
                    "cluster_id": int(labels[i]),
                    "assigned_at": now,
                },
            )

        # Пересчитываем cluster_task_stats из bandit_log
        await db.execute(sa.text("DELETE FROM cluster_task_stats"))
        await db.execute(
            sa.text("""
                INSERT INTO cluster_task_stats
                    (cluster_id, task_id, avg_reward, interaction_count, computed_at)
                SELECT
                    sc.cluster_id,
                    bl.task_id,
                    AVG(bl.reward)  AS avg_reward,
                    COUNT(*)        AS interaction_count,
                    :computed_at    AS computed_at
                FROM bandit_log bl
                JOIN student_clusters sc ON sc.student_id = bl.student_id
                WHERE bl.reward IS NOT NULL
                GROUP BY sc.cluster_id, bl.task_id
            """),
            {"computed_at": now},
        )

        await db.commit()

    logger.info(
        "clustering: %d students → %d clusters, centroids saved",
        len(student_ids),
        n_clusters,
    )


def assign_cluster_for_new_student(mastery: dict[str, float]) -> int | None:
    """
    Находит ближайший кластер для нового ученика по его mastery-вектору.
    Возвращает None если центроиды ещё не вычислены (первый запуск системы).
    """
    try:
        centroids = np.load(_CENTROIDS_PATH)
        kc_order = np.load(_KC_ORDER_PATH, allow_pickle=True)
    except FileNotFoundError:
        return None

    vector = np.array(
        [mastery.get(str(kc), 0.0) for kc in kc_order],
        dtype=np.float32,
    )
    distances = np.linalg.norm(centroids - vector, axis=1)
    return int(np.argmin(distances))


async def save_student_cluster(student_id: uuid.UUID, cluster_id: int) -> None:
    """Записывает cluster_id ученика в БД."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                INSERT INTO student_clusters (student_id, cluster_id, assigned_at)
                VALUES (:student_id, :cluster_id, :assigned_at)
                ON CONFLICT (student_id)
                DO UPDATE SET cluster_id = EXCLUDED.cluster_id,
                              assigned_at = EXCLUDED.assigned_at
            """),
            {
                "student_id": student_id,
                "cluster_id": cluster_id,
                "assigned_at": datetime.now(timezone.utc),
            },
        )
        await db.commit()
