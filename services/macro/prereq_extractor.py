"""
PrereqSubgraphExtractor — BFS назад по графу пре-реквизитов.

Принимает граф как dict (не делает HTTP-запросов).
Вызывающий код обязан передать граф уже загруженный из Graph-сервиса.

extract_prereq_subgraph:
  BFS от target_kc_id по обратным рёбрам PREREQUISITE.
  Включает KC у которых mastery < threshold - 0.05 (не освоены и не близко).
  Возвращает подграф {nodes: [kc_id, ...], edges: [{from, to, strength}]}.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass

MASTERY_GAP = 0.05  # KC считается "почти освоенной" если mastery >= threshold - gap


@dataclass(frozen=True)
class SubgraphEdge:
    from_kc: str     # пре-реквизит
    to_kc: str       # зависимая KC
    strength: float


def extract_prereq_subgraph(
    target_kc_id: str,
    mastery: dict[str, float],
    graph: dict[str, list[dict]],   # {kc_id: [{kc_id: prereq_id, strength: float}]}
    threshold: float = 0.75,
) -> dict:
    """
    BFS назад по графу пре-реквизитов от target_kc_id.

    Args:
        target_kc_id: целевая KC плана
        mastery: {kc_id: mastery_effective} ученика
        graph: {kc_id: [{kc_id: prereq_id, strength: float}]} — пре-реквизиты для каждой KC
        threshold: порог освоенности (например 0.75 для перехода к KC)

    Returns:
        {
          "nodes": [kc_id, ...],           — KC требующие работы (включая target)
          "edges": [{from, to, strength}]  — рёбра подграфа
        }
    """
    cutoff = threshold - MASTERY_GAP
    visited: set[str] = set()
    queue: deque[str] = deque([target_kc_id])
    nodes: list[str] = []
    edges: list[SubgraphEdge] = []

    while queue:
        kc_id = queue.popleft()
        if kc_id in visited:
            continue
        visited.add(kc_id)

        kc_mastery = mastery.get(kc_id, 0.0)
        # Включаем KC которая ещё не освоена (или это сама target)
        if kc_mastery < cutoff or kc_id == target_kc_id:
            nodes.append(kc_id)

        # BFS назад: смотрим пре-реквизиты этой KC
        for prereq in graph.get(kc_id, []):
            prereq_id = prereq["kc_id"]
            strength = float(prereq.get("strength", 0.5))
            prereq_mastery = mastery.get(prereq_id, 0.0)

            # Добавляем ребро в подграф
            edges.append(SubgraphEdge(from_kc=prereq_id, to_kc=kc_id, strength=strength))

            # Продолжаем BFS только по не-освоенным пре-реквизитам
            if prereq_mastery < cutoff and prereq_id not in visited:
                queue.append(prereq_id)

    return {
        "nodes": nodes,
        "edges": [{"from": e.from_kc, "to": e.to_kc, "strength": e.strength} for e in edges],
    }


def get_prereq_edge_map(subgraph: dict) -> dict[str, list[dict]]:
    """
    Из подграфа строит словарь {kc_id: [{from_kc, strength}]}.
    Удобно для быстрого lookup при вычислении reward.
    """
    result: dict[str, list[dict]] = {}
    for edge in subgraph["edges"]:
        result.setdefault(edge["to"], []).append(
            {"kc_id": edge["from"], "strength": edge["strength"]}
        )
    return result
