"""
Кластеризация учеников по mastery-векторам (GMM + BIC).

run_clustering() — запускается раз в час из cron:
  1. Читает все mastery-записи из БД
  2. Строит матрицу (студент × KC)
  3. Подбирает k через BIC (GaussianMixture, k от 3 до MAX_CLUSTERS)
  4. Записывает результаты в student_clusters
  5. Пересчитывает cluster_task_stats из bandit_log
  6. Сохраняет центроиды (GMM means) в PostgreSQL (cluster_centroids)

assign_cluster_for_new_student(mastery) — вызывается при создании ученика:
  Возвращает cluster_id по ближайшему центроиду. None если центроиды ещё не вычислены.
"""

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import sqlalchemy as sa
from sklearn.mixture import GaussianMixture

from shared.db import AsyncSessionLocal
from shared.config import clustering as _cfg

logger = logging.getLogger(__name__)

MAX_CLUSTERS = _cfg.N_CLUSTERS
MIN_CLUSTERS = 3

_centroids_cache: np.ndarray | None = None
_kc_order_cache: list[str] | None = None


async def _save_centroids_to_db(
    centroids: np.ndarray, kc_order: list[str], n_clusters: int
) -> None:
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                INSERT INTO cluster_centroids (id, n_clusters, kc_order, centroids_blob, computed_at)
                VALUES (1, :n_clusters, :kc_order, :blob, :now)
                ON CONFLICT (id) DO UPDATE
                    SET n_clusters = EXCLUDED.n_clusters,
                        kc_order = EXCLUDED.kc_order,
                        centroids_blob = EXCLUDED.centroids_blob,
                        computed_at = EXCLUDED.computed_at
            """),
            {
                "n_clusters": n_clusters,
                "kc_order": kc_order,
                "blob": centroids.astype(np.float32).tobytes(),
                "now": now,
            },
        )
        await db.commit()


async def _load_centroids_from_db() -> tuple[np.ndarray, list[str]] | None:
    global _centroids_cache, _kc_order_cache
    if _centroids_cache is not None and _kc_order_cache is not None:
        return _centroids_cache, _kc_order_cache

    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("SELECT n_clusters, kc_order, centroids_blob FROM cluster_centroids WHERE id = 1")
        )).fetchone()

    if not row:
        return None

    n_clusters, kc_order, blob = row[0], list(row[1]), bytes(row[2])
    n_features = len(kc_order)
    centroids = np.frombuffer(blob, dtype=np.float32).reshape(n_clusters, n_features).copy()

    _centroids_cache = centroids
    _kc_order_cache = kc_order
    return centroids, kc_order


def _select_n_clusters(matrix: np.ndarray, max_k: int) -> tuple[GaussianMixture, int]:
    """Select optimal k via BIC. Returns (best_model, best_k)."""
    max_k = min(max_k, matrix.shape[0])
    min_k = min(MIN_CLUSTERS, max_k)

    best_bic = float("inf")
    best_model = None
    best_k = min_k

    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=42)
        gmm.fit(matrix)
        bic = gmm.bic(matrix)
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_k = k

    return best_model, best_k


async def run_clustering() -> None:
    """Полный цикл кластеризации (GMM + BIC). Идемпотентен."""
    global _centroids_cache, _kc_order_cache

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            sa.text("SELECT student_id, kc_id, probability FROM mastery")
        )).fetchall()

    if not rows:
        logger.info("clustering: no mastery records, skipping")
        return

    student_ids = sorted({r[0] for r in rows})
    kc_ids = sorted({r[1] for r in rows})
    student_idx = {s: i for i, s in enumerate(student_ids)}
    kc_idx = {k: i for i, k in enumerate(kc_ids)}

    matrix = np.zeros((len(student_ids), len(kc_ids)), dtype=np.float32)
    for student_id, kc_id, prob in rows:
        matrix[student_idx[student_id], kc_idx[kc_id]] = float(prob)

    gmm, n_clusters = _select_n_clusters(matrix, MAX_CLUSTERS)
    labels = gmm.predict(matrix)

    await _save_centroids_to_db(gmm.means_.astype(np.float32), kc_ids, n_clusters)
    _centroids_cache = gmm.means_.astype(np.float32)
    _kc_order_cache = kc_ids

    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
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
        "clustering (GMM+BIC): %d students → %d clusters (max=%d), centroids stored in DB",
        len(student_ids),
        n_clusters,
        MAX_CLUSTERS,
    )


def assign_cluster_for_new_student(mastery: dict[str, float]) -> int | None:
    """
    Находит ближайший кластер для нового ученика по его mastery-вектору.
    Возвращает None если центроиды ещё не вычислены.

    Uses in-memory cache populated by run_clustering() or _load_centroids_from_db().
    For cold start (no cache), returns None — caller should retry after loading.
    """
    if _centroids_cache is None or _kc_order_cache is None:
        return None

    vector = np.array(
        [mastery.get(str(kc), 0.0) for kc in _kc_order_cache],
        dtype=np.float32,
    )
    distances = np.linalg.norm(_centroids_cache - vector, axis=1)
    return int(np.argmin(distances))


async def ensure_centroids_loaded() -> None:
    """Load centroids from DB into cache if not already loaded."""
    await _load_centroids_from_db()


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
