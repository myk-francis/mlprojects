import os
import pickle
from typing import Any
from api_logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "model_pipeline.pkl")

_model: Any = None

def load_model(path: str = MODEL_PATH) -> Any:
    global _model
    if _model is None:
        logger.info("Loading model from %s", path)
        with open(path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model loaded successfully")
    return _model

def get_model() -> Any:
    if _model is None:
        return load_model()
    return _model