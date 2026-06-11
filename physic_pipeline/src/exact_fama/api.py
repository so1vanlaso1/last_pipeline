from __future__ import annotations

from fastapi import FastAPI

from .pipeline import get_pipeline
from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="EXACT-FAMA Qwen3-8B Prototype", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return get_pipeline().predict(request)
