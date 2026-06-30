"""
Thompson Sampling — Bayesian contextual bandit for task selection.

Uses the same A (precision) and b (reward-weighted features) matrices as LinUCB.
Instead of deterministic UCB scoring, samples θ from the posterior:

    θ ~ N(A⁻¹b, v²A⁻¹)
    score = θᵀx

where v² controls exploration intensity (higher = more exploration).

Update rule (identical to LinUCB):
    A ← A + xxᵀ
    b ← b + r·x

Reference: Agrawal & Goyal (2013), "Thompson Sampling for Contextual Bandits
with Linear Payoffs"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.config import retrieval as _cfg

CONTEXT_DIM = _cfg.CONTEXT_DIM
V_SQUARED = _cfg.V_SQUARED


GLOBAL_CLUSTER_ID = -1


@dataclass
class ThompsonModel:
    kc_id: str
    cluster_id: int
    A: np.ndarray  # (CONTEXT_DIM × CONTEXT_DIM), float64 — precision matrix
    b: np.ndarray  # (CONTEXT_DIM,), float64
    _a_inv: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _mu: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def init(cls, kc_id: str, cluster_id: int = GLOBAL_CLUSTER_ID) -> ThompsonModel:
        return cls(
            kc_id=kc_id,
            cluster_id=cluster_id,
            A=np.eye(CONTEXT_DIM, dtype=np.float64),
            b=np.zeros(CONTEXT_DIM, dtype=np.float64),
        )

    @classmethod
    def from_bytes(cls, kc_id: str, cluster_id: int, a_bytes: bytes, b_bytes: bytes) -> ThompsonModel:
        A = np.frombuffer(a_bytes, dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM).copy()
        b = np.frombuffer(b_bytes, dtype=np.float64).copy()
        return cls(kc_id=kc_id, cluster_id=cluster_id, A=A, b=b)

    def to_bytes(self) -> tuple[bytes, bytes]:
        return self.A.astype(np.float64).tobytes(), self.b.astype(np.float64).tobytes()

    def _get_a_inv(self) -> np.ndarray:
        if self._a_inv is None:
            try:
                self._a_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self._a_inv = np.eye(CONTEXT_DIM, dtype=np.float64)
        return self._a_inv

    def _get_mu(self) -> np.ndarray:
        if self._mu is None:
            self._mu = self._get_a_inv() @ self.b
        return self._mu

    def score(self, x: np.ndarray) -> float:
        """Sample θ from posterior and return θᵀx."""
        x = x.astype(np.float64)
        mu = self._get_mu()
        cov = V_SQUARED * self._get_a_inv()
        try:
            theta = np.random.multivariate_normal(mu, cov)
        except np.linalg.LinAlgError:
            theta = mu
        return float(theta @ x)

    def score_exploit(self, x: np.ndarray) -> float:
        """Deterministic scoring (posterior mean) — for evaluation/logging."""
        x = x.astype(np.float64)
        return float(self._get_mu() @ x)

    def update(self, x: np.ndarray, reward: float) -> None:
        x = x.astype(np.float64)
        a_inv = self._get_a_inv()
        ax = a_inv @ x
        denom = 1.0 + float(x @ ax)
        if np.isfinite(denom) and denom > 1e-12:
            self._a_inv = a_inv - np.outer(ax, ax) / denom
        else:
            self._a_inv = None

        self.A += np.outer(x, x)
        self.b += reward * x
        self._mu = None
