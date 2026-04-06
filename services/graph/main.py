"""Сервис графа знаний."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.neo4j_client import neo4j_client
from .repository import GraphRepository
from .zpd import compute_zpd_candidates, KCNode, Prerequisite


# ---------------------------------------------------------------------------
# In-memory graph cache
# Граф меняется крайне редко — кешируем в RAM, избегая Neo4j на каждый /zpd.
# Инвалидация — через POST /cache/invalidate (вызывать после ресида графа).
# ---------------------------------------------------------------------------

_cache_kcs: list[KCNode] | None = None
_cache_prereqs: dict[str, list[Prerequisite]] | None = None


async def _get_graph_cached(repo: GraphRepository) -> tuple[list[KCNode], dict[str, list[Prerequisite]]]:
    global _cache_kcs, _cache_prereqs
    if _cache_kcs is None or _cache_prereqs is None:
        _cache_kcs = await repo.get_all_nodes()
        _cache_prereqs = await repo.get_all_prerequisites()
    return _cache_kcs, _cache_prereqs


@asynccontextmanager
async def lifespan(app: FastAPI):
    await neo4j_client.connect()
    yield
    await neo4j_client.close()


app = FastAPI(title="Graph Service", lifespan=lifespan)


def get_repo() -> GraphRepository:
    return GraphRepository(neo4j_client)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ZPDRequest(BaseModel):
    mastery: dict[str, float]   # kc_id → mastery_effective
    student_grade: int


class ZPDCandidateResponse(BaseModel):
    kc_id: str
    grade_introduced: int
    difficulty_base: float
    mastery_effective: float
    ready: bool
    subject: str


class PrerequisiteResponse(BaseModel):
    kc_id: str
    strength: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/nodes/{kc_id}")
async def get_node(kc_id: str):
    repo = get_repo()
    node = await repo.get_node(kc_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"KC '{kc_id}' not found")
    return node


@app.get("/nodes/{kc_id}/prerequisites", response_model=list[PrerequisiteResponse])
async def get_prerequisites(kc_id: str):
    repo = get_repo()
    all_kcs, all_prereqs = await _get_graph_cached(repo)
    kc_ids = {kc.kc_id for kc in all_kcs}
    if kc_id not in kc_ids:
        raise HTTPException(status_code=404, detail=f"KC '{kc_id}' not found")
    return all_prereqs.get(kc_id, [])


@app.post("/cache/invalidate", status_code=200)
async def invalidate_cache():
    """Сбросить кеш графа. Вызывать после обновления графа знаний."""
    global _cache_kcs, _cache_prereqs
    _cache_kcs = None
    _cache_prereqs = None
    return {"status": "invalidated"}


@app.post("/zpd", response_model=list[ZPDCandidateResponse])
async def get_zpd(req: ZPDRequest):
    """
    Принимает mastery ученика и его класс, возвращает KC-кандидатов в ZPD.
    Граф кешируется в RAM — Neo4j не опрашивается на каждый запрос.
    """
    repo = get_repo()
    all_kcs, all_prereqs = await _get_graph_cached(repo)

    candidates = compute_zpd_candidates(
        kcs=all_kcs,
        prerequisites=all_prereqs,
        mastery=req.mastery,
        student_grade=req.student_grade,
    )

    return [
        ZPDCandidateResponse(
            kc_id=c.kc_id,
            grade_introduced=c.grade_introduced,
            difficulty_base=c.difficulty_base,
            mastery_effective=c.mastery_effective,
            ready=len(c.unmet_prereqs) == 0,
            subject=c.subject,
        )
        for c in candidates
    ]


@app.get("/kcs")
async def get_all_kcs():
    """
    Возвращает все KC с grade_introduced и max_prereq_grade.
    max_prereq_grade — максимальный grade_introduced среди сильных пре-реквизитов KC
    (strength >= 0.5). Используется gateway для динамического cold_start prior.
    """
    repo = get_repo()
    all_kcs, all_prereqs = await _get_graph_cached(repo)
    kc_grade_map = {kc.kc_id: kc.grade_introduced for kc in all_kcs}
    result = []
    for kc in all_kcs:
        strong_prereqs = [p for p in all_prereqs.get(kc.kc_id, []) if p.strength >= 0.5]
        max_prereq_grade = max(
            (kc_grade_map.get(p.kc_id, 0) for p in strong_prereqs),
            default=0,
        )
        result.append({
            "kc_id": kc.kc_id,
            "grade_introduced": kc.grade_introduced,
            "max_prereq_grade": max_prereq_grade,
        })
    return result


@app.get("/path")
async def get_path(from_kc: str, to_kc: str):
    repo = get_repo()
    path = await repo.get_path(from_kc, to_kc)
    if not path:
        raise HTTPException(status_code=404, detail="Path not found")
    return {"path": path}
