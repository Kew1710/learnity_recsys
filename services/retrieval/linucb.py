"""
LinUCB — линейный контекстный бандит (upper confidence bound).

Одна модель на KC, общая для всех учеников.
Хранится в PostgreSQL: A (dim×dim) и b (dim) как BYTEA.

Оценка задания:
    score = θᵀx + α√(xᵀA⁻¹x)
    θ = A⁻¹b

Обновление после ответа:
    A ← A + xxᵀ
    b ← b + r·x
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.config import retrieval as _cfg

CONTEXT_DIM = _cfg.CONTEXT_DIM
ALPHA = _cfg.ALPHA


GLOBAL_CLUSTER_ID = -1   # запасная модель для студентов без кластера


@dataclass
class LinUCBModel:
    kc_id: str
    cluster_id: int
    A: np.ndarray  # (CONTEXT_DIM × CONTEXT_DIM), float64
    b: np.ndarray  # (CONTEXT_DIM,), float64
    _a_inv: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _theta: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def init(cls, kc_id: str, cluster_id: int = GLOBAL_CLUSTER_ID) -> "LinUCBModel":
        """Новая модель: A = I, b = 0."""
        return cls(
            kc_id=kc_id,
            cluster_id=cluster_id,
            A=np.eye(CONTEXT_DIM, dtype=np.float64),
            b=np.zeros(CONTEXT_DIM, dtype=np.float64),
        )

    @classmethod
    def from_bytes(cls, kc_id: str, cluster_id: int, a_bytes: bytes, b_bytes: bytes) -> "LinUCBModel":
        A = np.frombuffer(a_bytes, dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM).copy()
        b = np.frombuffer(b_bytes, dtype=np.float64).copy()
        return cls(kc_id=kc_id, cluster_id=cluster_id, A=A, b=b)

    def to_bytes(self) -> tuple[bytes, bytes]:
        return self.A.astype(np.float64).tobytes(), self.b.astype(np.float64).tobytes()

    def _get_a_inv(self) -> np.ndarray:
        """Ленивая инициализация A^-1; кэшируется между score/update."""
        if self._a_inv is None:
            try:
                self._a_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self._a_inv = np.eye(CONTEXT_DIM, dtype=np.float64)
        return self._a_inv

    def _get_theta(self) -> np.ndarray:
        """Ленивый расчёт theta = A^-1 b; инвалидация после каждого update."""
        if self._theta is None:
            self._theta = self._get_a_inv() @ self.b
        return self._theta

    def score(self, x: np.ndarray) -> float:
        """UCB-оценка для вектора признаков x."""
        x = x.astype(np.float64)
        A_inv = self._get_a_inv()
        theta = self._get_theta()
        ucb = float(np.sqrt(max(0.0, x @ A_inv @ x)))
        return float(theta @ x) + ALPHA * ucb

    def update(self, x: np.ndarray, reward: float) -> None:
        """Онлайн-обновление после получения награды."""
        x = x.astype(np.float64)

        # Обновляем обратную матрицу инкрементально:
        # (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        # Это убирает O(d^3) пересчёт inv(A) на каждом score().
        a_inv = self._get_a_inv()
        ax = a_inv @ x
        denom = 1.0 + float(x @ ax)
        if np.isfinite(denom) and denom > 1e-12:
            self._a_inv = a_inv - np.outer(ax, ax) / denom
        else:
            # Деградация численной устойчивости: пересчитаем при следующем score().
            self._a_inv = None

        self.A += np.outer(x, x)
        self.b += reward * x
        self._theta = None
