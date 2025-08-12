from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    prediction: List[float]
    success: bool
    detail: str | None = None