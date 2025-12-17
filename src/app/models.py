import datetime as dt

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Single observation for which we want to forecast sales."""

    date: dt.date = Field(..., description="Дата в формате YYYY-MM-DD")
    store: int = Field(..., ge=1, le=10, description="ID магазина (1–10)")
    item: int = Field(..., ge=1, le=50, description="ID товара (1–50)")


class PredictionRequest(BaseModel):
    """Batch of observations for inference."""

    instances: list[PredictionItem]


class PredictionResult(BaseModel):
    """Single prediction result."""

    date: dt.date
    store: int
    item: int
    prediction: float


class PredictionResponse(BaseModel):
    """Batch of prediction results."""

    predictions: list[PredictionResult]


