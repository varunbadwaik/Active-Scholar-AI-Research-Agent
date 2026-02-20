"""Active Scholar — FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from graph import graph
from quick_report import generate_quick_report
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
    """Background task that runs the research pipeline."""
    mode = request.mode
    logger.info("Job %s started — topic: %s, mode: %s", job_id, request.topic, mode)
    try:
        if mode == "quick":
            # Fast path: Tavily search + 1 LLM call
            result = await generate_quick_report(
                topic=request.topic,
                scope=request.scope,
                max_results=5,
            )
            _jobs[job_id] = result  # Store as plain dict
        else:
            # Full deep research via LangGraph
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

        # --- Save report to disk ---
        report_md = (
            result.get("report", "") if isinstance(result, dict)
            else result.get("report", "")
        )

        import re as _re
        from pathlib import Path

        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        safe_topic = _re.sub(r"[^a-zA-Z0-9_\-]", "_", request.topic)[:50]
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{safe_topic}.md"
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info("Job %s completed — saved to %s", job_id, file_path)

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
    Supports ``mode``: "quick" (fast, 30s) or "deep" (thorough, 3-10min).
    """
    job_id = str(uuid4())
    _jobs[job_id] = "running"
    background_tasks.add_task(_run_investigation, job_id, request)
    return {"job_id": job_id, "status": "running", "mode": request.mode}


@app.get("/research/{job_id}")
async def get_result(job_id: str):
    """Poll for the result of a research investigation."""
    result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if isinstance(result, str):
        # Still running or errored
        return {"job_id": job_id, "status": result}
    if isinstance(result, dict):
        # Quick mode result — already a dict
        result["status"] = "done"
        return result
    # Deep mode result — Pydantic model
    data = result.model_dump()
    data["status"] = "done"
    return data


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def serve_frontend():
    """Serve the web frontend."""
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")


@app.get("/reports")
async def list_reports():
    """List all saved reports."""
    from pathlib import Path
    output_dir = Path("reports")
    if not output_dir.exists():
        return []
    
    reports = []
    for f in sorted(output_dir.glob("*.md"), reverse=True):
        reports.append({
            "filename": f.name,
            "created_at": f.stat().st_mtime,
            "size": f.stat().st_size
        })
    return reports


@app.get("/reports/{filename}")
async def get_report_content(filename: str):
    """Get the content of a specific report."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    file_path = Path("reports") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    # Return as text/markdown
    return FileResponse(file_path, media_type="text/markdown")
