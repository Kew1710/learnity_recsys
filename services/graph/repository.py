"""Neo4j операции для сервиса графа."""

from __future__ import annotations
from shared.neo4j_client import Neo4jClient
from .zpd import KCNode, Prerequisite


class GraphRepository:
    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    # ---------------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------------

    async def get_node(self, kc_id: str) -> dict | None:
        rows = await self.client.query(
            "MATCH (kc:KC {kc_id: $kc_id}) RETURN kc",
            {"kc_id": kc_id},
        )
        return rows[0]["kc"] if rows else None

    async def get_all_nodes(self) -> list[KCNode]:
        rows = await self.client.query("MATCH (kc:KC) RETURN kc")
        return [
            KCNode(
                kc_id=r["kc"]["kc_id"],
                grade_introduced=r["kc"]["grade_introduced"],
                difficulty_base=r["kc"]["difficulty_base"],
                half_life_days=r["kc"].get("half_life_days", 30.0),
                subject=r["kc"].get("subject", ""),
            )
            for r in rows
        ]

    # ---------------------------------------------------------------------------
    # Edges
    # ---------------------------------------------------------------------------

    async def get_prerequisites(self, kc_id: str) -> list[dict]:
        """Возвращает список пререквизитов для одной KC."""
        rows = await self.client.query(
            """
            MATCH (prereq:KC)-[r:PREREQUISITE]->(kc:KC {kc_id: $kc_id})
            RETURN prereq.kc_id AS kc_id, r.strength AS strength
            """,
            {"kc_id": kc_id},
        )
        return rows

    async def get_all_prerequisites(self) -> dict[str, list[Prerequisite]]:
        """Возвращает все рёбра графа: kc_id → [Prerequisite]."""
        rows = await self.client.query(
            """
            MATCH (prereq:KC)-[r:PREREQUISITE]->(kc:KC)
            RETURN kc.kc_id AS target, prereq.kc_id AS source, r.strength AS strength
            """
        )
        result: dict[str, list[Prerequisite]] = {}
        for row in rows:
            target = row["target"]
            result.setdefault(target, []).append(
                Prerequisite(kc_id=row["source"], strength=row["strength"])
            )
        return result

    async def get_path(self, from_kc: str, to_kc: str) -> list[str]:
        """Кратчайший путь между двумя KC по рёбрам пререквизитов."""
        rows = await self.client.query(
            """
            MATCH path = shortestPath(
              (a:KC {kc_id: $from_kc})-[:PREREQUISITE*]->(b:KC {kc_id: $to_kc})
            )
            RETURN [node IN nodes(path) | node.kc_id] AS path
            """,
            {"from_kc": from_kc, "to_kc": to_kc},
        )
        return rows[0]["path"] if rows else []

    # ---------------------------------------------------------------------------
    # Seed helpers
    # ---------------------------------------------------------------------------

    async def clear(self) -> None:
        await self.client.execute("MATCH (n) DETACH DELETE n")

    async def create_kc(self, props: dict) -> None:
        await self.client.execute(
            "CREATE (kc:KC $props)",
            {"props": props},
        )

    async def create_prerequisite(self, from_kc: str, to_kc: str, strength: float) -> None:
        await self.client.execute(
            """
            MATCH (a:KC {kc_id: $from_kc}), (b:KC {kc_id: $to_kc})
            CREATE (a)-[:PREREQUISITE {strength: $strength}]->(b)
            """,
            {"from_kc": from_kc, "to_kc": to_kc, "strength": strength},
        )
