"""HTTP microservice for the Semantic Product Search Engine.

Endpoints:
    GET  /health         liveness probe
    POST /tag            tag on create (flow 1)
    POST /search         realtime search (flow 2)
    POST /rebuild        kick off a background catalog rebuild

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

The TaggingService (and the ~2.2 GB bge-m3 model) is built lazily on first use,
so startup is fast; the first /tag or /search request pays the model-load cost.
Call POST /rebuild (or GET /health?warm=1) once at boot to warm it if you prefer.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from semantic_search.service import TaggingService

app = FastAPI(title="Semantic Product Search Engine", version="0.1.0")

_service: Optional[TaggingService] = None


def service() -> TaggingService:
    global _service
    if _service is None:
        _service = TaggingService()
    return _service


class TagRequest(BaseModel):
    name: str
    description: str = ""
    external_id: Optional[str] = None
    write_source_tags: bool = True


class TagResponse(BaseModel):
    external_id: Optional[str]
    category: Optional[str]
    tags: List[str]
    discovered_tags: List[str]


class SearchRequest(BaseModel):
    query: str
    k: int = 10


@app.get("/health")
def health(warm: bool = False):
    if warm:
        service()  # build + load the model now
    return {"status": "ok", "model_loaded": _service is not None}


@app.post("/tag", response_model=TagResponse)
def tag(body: TagRequest) -> TagResponse:
    rec = service().tag_product(
        body.name,
        body.description,
        external_id=body.external_id,
        write_source_tags=body.write_source_tags,
    )
    return TagResponse(
        external_id=rec["external_id"],
        category=rec["category"],
        tags=rec["tags"],
        discovered_tags=rec["discovered_tags"],
    )


@app.post("/search")
def search(body: SearchRequest):
    return {"query": body.query, "results": service().search(body.query, body.k)}


@app.post("/rebuild")
def rebuild(background: BackgroundTasks, since: Optional[str] = None):
    """Start a catalog rebuild in the background. Returns immediately."""
    background.add_task(service().rebuild, since=since)
    return {"status": "started", "since": since}
