import os
from neo4j import AsyncGraphDatabase, AsyncDriver

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "learnity123")


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        assert self._driver, "Not connected"
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters or {})
            return [record.data() async for record in result]

    async def execute(self, cypher: str, parameters: dict | None = None) -> None:
        assert self._driver, "Not connected"
        async with self._driver.session() as session:
            await session.run(cypher, parameters or {})

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


# Singleton для использования в FastAPI lifespan
neo4j_client = Neo4jClient()
