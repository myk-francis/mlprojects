from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from contextlib import asynccontextmanager
from model_loader import  get_model
from schema import  PredictResponse
from api_logger import get_logger
from predictor import PredictPipeline, CustomeData, TrainRequest
from src.components.data_ingestion import DataIngestion

import pandas as pd
from pydantic import BaseModel
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logger = get_logger(__name__)

pipeline = PredictPipeline()  # load model once at startup
data_ingestion = DataIngestion()

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
       

        preds = pipeline.predict(data.get_data_as_dataframe())
        return PredictResponse(prediction=preds.tolist(), success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/train")
def train_api(request: TrainRequest):
    try:
        # Convert input list of dicts to DataFrame
        # Convert list of TrainingRecord to DataFrame
        new_data_df = pd.DataFrame([r.model_dump() for r in request.data])

        # Ensure columns are exactly what we expect
        expected_columns = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course",
            "math_score", "reading_score", "writing_score"
        ]
        missing = set(expected_columns) - set(new_data_df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Append new data to CSV
        data_ingestion.append_data(new_data_df)

        # Retrain model on full data
        train_data,test_data = data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

        modeltrainer=ModelTrainer()
        modeltrainer.initiate_model_trainer(train_arr,test_arr)

        return {"success": True, "detail": f"Model retrained on new data with {len(new_data_df)} records"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Training error")
        raise HTTPException(status_code=500, detail=str(e))
    
