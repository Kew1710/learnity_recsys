"""
Точка входа для cron-задачи кластеризации.

Запуск (из корня проекта):
    python -m tools.cron_cluster

Добавить в crontab:
    0 * * * * cd /path/to/learnity && python -m tools.cron_cluster >> logs/cluster.log 2>&1
"""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from services.clustering.cluster import run_clustering


if __name__ == "__main__":
    asyncio.run(run_clustering())
