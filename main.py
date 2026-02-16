"""Active Scholar — FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException

from graph import graph
from state import (
    ReportMetadata,
    ResearchReport,
    ResearchRequest,
    ResearchState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Active Scholar",
    description="Autonomous research agent that searches, retrieves, "
    "analyses, and synthesises information into structured reports.",
    version="1.0.0",
)

# In-memory job store (swap for Redis / DB in production)
_jobs: dict[str, ResearchReport | str] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_initial_state(req: ResearchRequest) -> ResearchState:
    return ResearchState(
        topic=req.topic,
        scope=req.scope,
        constraints=req.constraints,
        max_search_rounds=req.max_search_rounds,
        search_queries=[],
        search_results=[],
        explored_urls=set(),
        current_search_round=0,
        search_exhausted=False,
        ingested_chunks=[],
        retrieval_results=[],
        claims=[],
        contradictions=[],
        evidence_quality={},
        synthesis_complete=False,
        needs_more_info=False,
        follow_up_queries=[],
        report=None,
        report_metadata=None,
    )


async def _run_investigation(job_id: str, request: ResearchRequest) -> None:
    """Background task that runs the full LangGraph pipeline."""
    logger.info("Job %s started — topic: %s", job_id, request.topic)
    try:
        state = _build_initial_state(request)
        result = await graph.ainvoke(state)

        meta_raw = result.get("report_metadata", {})
        report = ResearchReport(
            report_markdown=result.get("report", ""),
            metadata=ReportMetadata(**meta_raw),
            claims=result.get("claims", []),
            contradictions=result.get("contradictions", []),
            sources=result.get("search_results", []),
        )
        _jobs[job_id] = report
        logger.info("Job %s completed successfully", job_id)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = f"error: {exc}"


# ── Routes ───────────────────────────────────────────────────────────────────


@app.post("/research", response_model=dict, status_code=202)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
):
    """Submit a new research investigation.

    Returns a ``job_id`` that can be polled via ``GET /research/{job_id}``.
    """
    job_id = str(uuid4())
    _jobs[job_id] = "running"
    background_tasks.add_task(_run_investigation, job_id, request)
    return {"job_id": job_id, "status": "running"}


@app.get("/research/{job_id}")
async def get_result(job_id: str):
    """Poll for the result of a research investigation."""
    result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if isinstance(result, str):
        # Still running or errored
        return {"job_id": job_id, "status": result}
    return result.model_dump()


@app.get("/health")
async def health():
    return {"status": "ok"}
