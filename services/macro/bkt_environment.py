"""
BKTEnvironment — симулятор RL-среды для Macro Planner (Режим 1).

Каждый шаг среды = одна попытка ученика на задании для выбранной KC.
Симулирует BKT-обновление mastery и считает shaped reward.

Reward:
  Δmastery(target) + 0.1 × Σ Δmastery(prereq) × edge_strength − 0.01
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

from services.profile.bkt import bkt_posterior, apply_transit

# Дефолтные BKT параметры для симуляции
DEFAULT_P_TRANSIT = 0.10
DEFAULT_P_SLIP = 0.10
DEFAULT_P_GUESS = 0.20

# За один макро-шаг симулируем N микро-попыток
MICRO_STEPS_PER_MACRO = 3

# Масштаб отрицательной награды за каждый шаг (стоимость времени)
TIME_COST = 0.01


@dataclass
class BKTState:
    mastery: dict[str, float]     # kc_id → текущий mastery
    steps_taken: int = 0


class BKTEnvironment:
    """
    Симулятор для обучения RL-агента (SubgraphQAgent).

    Принимает подграф как dict — без HTTP/DB зависимостей.
    """

    def __init__(
        self,
        subgraph: dict,
        target_kc_id: str,
        target_mastery: float = 0.80,
        bkt_params: dict[str, dict] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """
        Args:
            subgraph: выход extract_prereq_subgraph — {nodes, edges}
            target_kc_id: KC которую хотим освоить
            target_mastery: порог считающийся "освоено"
            bkt_params: {kc_id: {p_transit, p_slip, p_guess}} — опциональные индивидуальные параметры
            rng: для воспроизводимости тестов
        """
        self.subgraph = subgraph
        self.target_kc_id = target_kc_id
        self.target_mastery = target_mastery
        self.bkt_params = bkt_params or {}
        self.rng = rng or random.Random()

        # Строим map {kc_id: [{kc_id: prereq_id, strength}]} для reward
        self._prereq_map: dict[str, list[dict]] = {}
        for edge in subgraph["edges"]:
            self._prereq_map.setdefault(edge["to"], []).append(
                {"kc_id": edge["from"], "strength": edge["strength"]}
            )

        self._initial_mastery: dict[str, float] = {}
        self._state: BKTState = BKTState(mastery={})

    def reset(self, initial_mastery: dict[str, float]) -> BKTState:
        """Инициализирует среду с начальным mastery. Возвращает начальный state."""
        self._initial_mastery = deepcopy(initial_mastery)
        self._state = BKTState(mastery=deepcopy(initial_mastery))
        return deepcopy(self._state)

    def step(self, kc_id: str) -> tuple[BKTState, float, bool]:
        """
        Один макро-шаг: симулировать MICRO_STEPS_PER_MACRO BKT попыток для kc_id.

        Returns:
            (new_state, reward, done)
        """
        mastery_before = deepcopy(self._state.mastery)

        for _ in range(MICRO_STEPS_PER_MACRO):
            self._simulate_attempt(kc_id)

        mastery_after = self._state.mastery
        reward = self._compute_reward(mastery_before, mastery_after)
        self._state.steps_taken += 1

        done = mastery_after.get(self.target_kc_id, 0.0) >= self.target_mastery

        return deepcopy(self._state), reward, done

    def _simulate_attempt(self, kc_id: str) -> None:
        """Симулирует одну попытку ученика: score ~ Bernoulli(mastery) + BKT update."""
        m = self._state.mastery.get(kc_id, 0.0)
        params = self.bkt_params.get(kc_id, {})
        p_slip = params.get("p_slip", DEFAULT_P_SLIP)
        p_guess = params.get("p_guess", DEFAULT_P_GUESS)
        p_transit = params.get("p_transit", DEFAULT_P_TRANSIT)

        # Симулируем ответ ученика: правильный с вероятностью mastery*(1-slip) + (1-mastery)*guess
        p_correct = m * (1 - p_slip) + (1 - m) * p_guess
        score = 1.0 if self.rng.random() < p_correct else 0.0

        # BKT update
        posterior = bkt_posterior(m, score, p_slip, p_guess)
        new_mastery = apply_transit(posterior, p_transit)
        self._state.mastery[kc_id] = min(1.0, new_mastery)

    def _compute_reward(
        self,
        mastery_before: dict[str, float],
        mastery_after: dict[str, float],
    ) -> float:
        """
        R = Δmastery(target) + 0.1 × Σ Δmastery(prereq) × edge_strength − TIME_COST
        """
        delta_target = (
            mastery_after.get(self.target_kc_id, 0.0)
            - mastery_before.get(self.target_kc_id, 0.0)
        )

        prereq_bonus = 0.0
        for edge in self.subgraph["edges"]:
            prereq_id = edge["from"]
            strength = edge["strength"]
            delta_prereq = (
                mastery_after.get(prereq_id, 0.0)
                - mastery_before.get(prereq_id, 0.0)
            )
            prereq_bonus += delta_prereq * strength

        return delta_target + 0.1 * prereq_bonus - TIME_COST
