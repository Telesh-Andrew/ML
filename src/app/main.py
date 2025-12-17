from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from .predictor import SalesPredictor


app = FastAPI(
    title="Store Item Demand Forecasting API",
    version="1.0.0",
    description="FastAPI service that wraps the trained LightGBM model.",
)

# Инициализируем предиктор один раз на процесс
predictor = SalesPredictor()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return forecasts for a batch of (date, store, item) triples."""
    if not request.instances:
        raise HTTPException(status_code=400, detail="`instances` must not be empty")

    try:
        predictions = predictor.predict(request.instances)
    except Exception as exc:  # pragma: no cover - защитный блок
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[PredictionResult] = []
    for item, value in zip(request.instances, predictions):
        results.append(
            PredictionResult(
                date=item.date,
                store=item.store,
                item=item.item,
                prediction=float(value),
            )
        )

    return PredictionResponse(predictions=results)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


