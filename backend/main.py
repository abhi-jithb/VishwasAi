"""FastAPI server for Vishwas AI evaluation API."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from backend.evaluation.pipeline import evaluate

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Vishwas AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    retrieval: bool = True
    inspect: bool = False

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("answer")
    @classmethod
    def answer_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer must not be empty")
        if len(v) > 3000:
            raise ValueError("answer must be under 3000 characters")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest) -> Any:
    """Run the full Vishwas AI evaluation pipeline on a (question, answer) pair."""
    try:
        result = evaluate(
            question=req.question,
            answer=req.answer,
            retrieval=req.retrieval,
            inspect=req.inspect,
        )
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Static files – mount AFTER API routes so /api/* hits routes first
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="static")
