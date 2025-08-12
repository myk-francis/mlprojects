from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from contextlib import asynccontextmanager
from model_loader import  get_model
from schema import  PredictResponse
from api_logger import get_logger
from predictor import PredictPipeline, CustomeData

logger = get_logger(__name__)

pipeline = PredictPipeline()  # load model once at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # load_model()  # load model at startup
        logger.info("Model loaded during lifespan startup")
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)
    yield
    # Place shutdown logic here if needed
    logger.info("App shutdown")

app = FastAPI(
    title="MLProjects - Model API",
    version="0.1",
    description="Prediction API for MLProjects model pipeline",
    lifespan=lifespan
)

# Optional: enable CORS if you have a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
def health():
    """Simple health check."""
    try:
        _ = get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "ok", "model_loaded": False}

@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(data: CustomeData):
    try:
        # model = get_model()
        # if model is None:
        #     raise HTTPException(status_code=500, detail="Model not loaded")

        # features = np.array(request.features, dtype=float)
        # if features.ndim == 1:
        #     features = features.reshape(1, -1)

        # preds = model.predict(features)

        # return PredictResponse(prediction=preds.tolist(), success=True)

        preds = pipeline.predict(data.get_data_as_dataframe())
        return PredictResponse(prediction=preds.tolist(), success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))