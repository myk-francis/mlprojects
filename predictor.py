# src/predictor.py
import os
import sys
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from src.exception import CustomException  # adjust import path
from src.utils import load_object  # adjust import path
from threading import Lock
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

class TrainingRecord(BaseModel):
    gender: Literal["male", "female"]
    race_ethnicity: Literal["group A", "group B", "group C", "group D", "group E"]
    parental_level_of_education: str
    lunch: Literal["standard", "free/reduced"]
    test_preparation_course: Literal["none", "completed"]
    math_score: int = Field(..., ge=0, le=100)
    reading_score: int = Field(..., ge=0, le=100)
    writing_score: int = Field(..., ge=0, le=100)

class TrainRequest(BaseModel):
    data: List[TrainingRecord]
    target_column: str = "math_score"  # assuming math_score is target for training


class CustomeData(BaseModel):
    gender: Literal["male", "female"]
    race_ethnicity: Literal[
        "group A", "group B", "group C", "group D", "group E"
    ]  # adjust categories to your dataset
    parental_level_of_education: str
    lunch: Literal["standard", "free/reduced"]
    test_preparation_course: Literal["none", "completed"]
    reading_score: float = Field(..., ge=0, le=100)
    writing_score: float = Field(..., ge=0, le=100)

    def get_data_as_dataframe(self):
        try:
            data_dict = self.model_dump()
            # wrap each value in a list to create one-row DataFrame
            return pd.DataFrame({k: [v] for k, v in data_dict.items()})
        except Exception as e:
            raise CustomException(e, sys.exc_info())


class PredictPipeline:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            self.preprocessor = load_object(file_path=preprocessor_path)
            self.model = load_object(file_path=model_path)
        except Exception as e:
            raise CustomException(f"Model loading failed: {e}", sys.exc_info())

    def predict(self, features: pd.DataFrame):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
    
    
