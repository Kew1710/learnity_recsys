"""Тесты PrereqSubgraphExtractor."""
import pytest
from services.macro.prereq_extractor import extract_prereq_subgraph, get_prereq_edge_map


# Простой граф: A → B → C (B пре-реквизит C, A пре-реквизит B)
SIMPLE_GRAPH = {
    "C": [{"kc_id": "B", "strength": 0.9}],
    "B": [{"kc_id": "A", "strength": 0.8}],
    "A": [],
}


class TestExtractPrereqSubgraph:
    def test_target_always_included(self):
        mastery = {"A": 0.9, "B": 0.9, "C": 0.9}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH)
        assert "C" in result["nodes"]

    def test_already_mastered_prereqs_excluded(self):
        # A и B уже освоены (0.9 >= 0.70 = threshold - 0.05)
        mastery = {"A": 0.9, "B": 0.9, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH, threshold=0.75)
        assert "A" not in result["nodes"]
        assert "B" not in result["nodes"]
        assert "C" in result["nodes"]

    def test_unmastered_prereqs_included(self):
        mastery = {"A": 0.2, "B": 0.3, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH, threshold=0.75)
        assert "A" in result["nodes"]
        assert "B" in result["nodes"]
        assert "C" in result["nodes"]

    def test_edges_captured(self):
        mastery = {"A": 0.2, "B": 0.3, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH)
        edge_pairs = {(e["from"], e["to"]) for e in result["edges"]}
        assert ("B", "C") in edge_pairs
        assert ("A", "B") in edge_pairs

    def test_no_edges_for_mastered_chain(self):
        # Если A освоена, BFS не продолжается за A
        mastery = {"A": 0.9, "B": 0.3, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH, threshold=0.75)
        # B включён (не освоен), A нет
        assert "B" in result["nodes"]
        assert "A" not in result["nodes"]

    def test_diamond_graph_no_duplicates(self):
        # A → B, A → C, B → D, C → D (diamond)
        diamond = {
            "D": [{"kc_id": "B", "strength": 0.8}, {"kc_id": "C", "strength": 0.8}],
            "B": [{"kc_id": "A", "strength": 0.9}],
            "C": [{"kc_id": "A", "strength": 0.9}],
            "A": [],
        }
        mastery = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.0}
        result = extract_prereq_subgraph("D", mastery, diamond)
        # Каждый узел должен быть только один раз
        assert len(result["nodes"]) == len(set(result["nodes"]))
        assert "A" in result["nodes"]

    def test_empty_graph_returns_only_target(self):
        result = extract_prereq_subgraph("X", {}, {})
        assert result["nodes"] == ["X"]
        assert result["edges"] == []

    def test_threshold_boundary(self):
        # threshold=0.75, cutoff=0.70
        # mastery=0.70 → ровно на границе → не включается (< cutoff False → не работает)
        mastery = {"B": 0.70, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH, threshold=0.75)
        # 0.70 < 0.70 — False → B НЕ включается
        assert "B" not in result["nodes"]

    def test_threshold_just_below(self):
        mastery = {"B": 0.69, "C": 0.0}
        result = extract_prereq_subgraph("C", mastery, SIMPLE_GRAPH, threshold=0.75)
        assert "B" in result["nodes"]


class TestGetPrereqEdgeMap:
    def test_builds_correct_map(self):
        subgraph = {
            "edges": [
                {"from": "A", "to": "B", "strength": 0.8},
                {"from": "B", "to": "C", "strength": 0.9},
            ]
        }
        edge_map = get_prereq_edge_map(subgraph)
        assert "B" in edge_map
        assert edge_map["B"][0]["kc_id"] == "A"
        assert "C" in edge_map
        assert edge_map["C"][0]["kc_id"] == "B"

    def test_empty_edges(self):
        edge_map = get_prereq_edge_map({"edges": []})
        assert edge_map == {}
