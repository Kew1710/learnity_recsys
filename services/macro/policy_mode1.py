"""
SubgraphQAgent — Q-learning агент для Macro Planner (Режим 1).

Учится строить оптимальную последовательность KC для достижения target_mastery.
Состояние — дискретизированный mastery-профиль подграфа.
Действие — выбор KC для следующего задания.

train_policy:
  Обучает агента на BKTEnvironment за n_episodes эпизодов.
  Сохраняет Q-таблицу в models/ для повторного использования.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field

from .bkt_environment import BKTEnvironment


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

MASTERY_BINS = 5          # дискретизация: 0.0..0.2, 0.2..0.4, ..., 0.8..1.0
MAX_STEPS_PER_EPISODE = 50  # ограничение длины эпизода


def _discretize(mastery: float) -> int:
    """Mastery → bin [0..MASTERY_BINS-1]."""
    return min(MASTERY_BINS - 1, int(mastery * MASTERY_BINS))


def _state_key(mastery: dict[str, float], node_order: list[str]) -> tuple:
    """
    Детерминированный ключ состояния из mastery-профиля.
    node_order фиксирует порядок KC → воспроизводимый tuple.
    """
    return tuple(_discretize(mastery.get(kc, 0.0)) for kc in node_order)


# ---------------------------------------------------------------------------
# Q-Agent
# ---------------------------------------------------------------------------

class SubgraphQAgent:
    """
    Табличный Q-learning для фиксированного подграфа.

    Q-таблица: state_key → {kc_id: q_value}
    """

    def __init__(
        self,
        node_order: list[str],
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        rng: random.Random | None = None,
    ) -> None:
        self.node_order = node_order
        self.lr = learning_rate
        self.gamma = gamma
        self.rng = rng or random.Random()
        self.q: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def select_action(
        self,
        state: dict[str, float],
        available_actions: list[str],
        epsilon: float = 0.1,
    ) -> str:
        """ε-greedy выбор KC."""
        if not available_actions:
            raise ValueError("Список доступных KC пуст")
        if self.rng.random() < epsilon:
            return self.rng.choice(available_actions)
        key = _state_key(state, self.node_order)
        q_row = self.q[key]
        return max(available_actions, key=lambda a: q_row[a])

    def update(
        self,
        state: dict[str, float],
        action: str,
        reward: float,
        next_state: dict[str, float],
        available_next: list[str],
    ) -> None:
        """Bellman update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]"""
        s_key = _state_key(state, self.node_order)
        ns_key = _state_key(next_state, self.node_order)

        q_current = self.q[s_key][action]

        if available_next:
            q_next_max = max(self.q[ns_key][a] for a in available_next)
        else:
            q_next_max = 0.0

        self.q[s_key][action] = q_current + self.lr * (
            reward + self.gamma * q_next_max - q_current
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(
    subgraph: dict,
    target_kc_id: str,
    initial_mastery: dict[str, float],
    target_mastery: float = 0.80,
    n_episodes: int = 2000,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    bkt_params: dict | None = None,
    rng: random.Random | None = None,
) -> SubgraphQAgent:
    """
    Обучает Q-агента на BKTEnvironment.

    Args:
        subgraph: {nodes, edges} из PrereqSubgraphExtractor
        target_kc_id: KC которую нужно освоить
        initial_mastery: начальный mastery кластера
        target_mastery: порог освоения
        n_episodes: количество эпизодов обучения
        epsilon_start/end: линейный decay exploration rate
        bkt_params: параметры BKT для симулятора

    Returns:
        обученный SubgraphQAgent
    """
    _rng = rng or random.Random()
    node_order = sorted(subgraph["nodes"])
    actions = node_order  # действия = KC в подграфе

    agent = SubgraphQAgent(node_order=node_order, rng=_rng)
    env = BKTEnvironment(
        subgraph=subgraph,
        target_kc_id=target_kc_id,
        target_mastery=target_mastery,
        bkt_params=bkt_params,
        rng=_rng,
    )

    for episode in range(n_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / n_episodes)
        state = env.reset(initial_mastery)

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state.mastery, actions, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state.mastery, action, reward, next_state.mastery, actions)
            state = next_state
            if done:
                break

    return agent


def save_policy(agent: SubgraphQAgent, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(agent, f)


def load_policy(path: str) -> SubgraphQAgent:
    with open(path, "rb") as f:
        return pickle.load(f)


def policy_path(cluster_id: int, target_kc: str) -> str:
    return f"models/policy_cluster_{cluster_id}_{target_kc}.pkl"
