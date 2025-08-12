import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def append_data(self, new_data_df: pd.DataFrame):
        try:
            if os.path.exists('notebook/stud.csv'):
                # Load existing data
                existing_df = pd.read_csv('notebook/stud.csv')
                # Append new data
                combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            else:
                # No existing data, so new data is the combined data
                combined_df = new_data_df

            # Save combined data back to CSV
            combined_df.to_csv('notebook/stud.csv', index=False)
            print(f"Data appended successfully. Total records: {len(combined_df)}")
        except Exception as e:
            raise Exception(f"Error appending data: {e}")

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv('notebook/stud.csv')
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys.exc_info())
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))