"""
Seed-данные: методический граф по математике 5–11 класс.

Данные KC-нод и рёбер вынесены в kc_data.py (без внешних зависимостей).
Этот модуль добавляет half_life_days и выполняет загрузку в Neo4j.

Запуск: python -m services.graph.seed
"""

import asyncio
from shared.neo4j_client import Neo4jClient
from .repository import GraphRepository
from .kc_data import KCS, EDGES

_HALF_LIFE_BY_SUBJECT = {
    "arithmetic": 90,
    "algebra":    45,
    "geometry":   60,
    "statistics": 30,
}
for _kc in KCS:
    _kc["half_life_days"] = _HALF_LIFE_BY_SUBJECT.get(_kc["subject"], 45)


async def run_seed(clear: bool = True) -> None:
    async with Neo4jClient() as client:
        repo = GraphRepository(client)

        if clear:
            await repo.clear()
            print("Граф очищен.")

        for kc in KCS:
            await repo.create_kc(kc)
        print(f"Создано {len(KCS)} KC-нод.")

        for from_kc, to_kc, strength in EDGES:
            await repo.create_prerequisite(from_kc, to_kc, strength)
        print(f"Создано {len(EDGES)} рёбер.")


if __name__ == "__main__":
    asyncio.run(run_seed())
