import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTraining
from src.components.model_trainer import ModelTrainingConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts/data_ingestion',"train.csv")
    test_data_path: str=os.path.join('artifacts/data_ingestion',"test.csv")
    raw_data_path: str=os.path.join('artifacts/data_ingestion',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv('notebook/data/student.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df.drop(columns=['Unnamed: 0','Id'],inplace=True)
            df.dropna(inplace=True)
            df['Additional_Work'] = df['Additional_Work'].map({'Yes': 0, 'No' : 1})
            df.drop(df[df['Attendance'] == '3'].index,inplace=True)
            df.drop(df[df['Listening_in_Class'] == '6'].index,inplace=True)
            df['Sports_activity'] = df['Sports_activity'].map({'Yes': 1, 'No' : 0})
            df['Listening_in_Class'] = df['Listening_in_Class'].map({'Yes': 1, 'No' : 0})
            df['Project_work'] = df['Project_work'].map({'Yes': 1, 'No' : 0})
            df['Notes'] = df['Notes'].map({'Yes': 1, 'No' : 0})
            df['Reading'] = df['Reading'].map({'Yes': 1, 'No' : 0})
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female' : 0})
            df['Transportation'] = df['Transportation'].map({'Private': 1, 'Bus' : 0})
            df['Attendance'] = df['Attendance'].map({'Always': 3, 'Sometimes' : 2, 'Never' : 1})
            df['Grade'] = df['Grade'].map({'AA': 1, 'BA' : 2, 'BB' : 3, 'CC' : 4, 'DD' : 5, 'DC' : 6, 'CB' : 7, 'Fail':0})

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size= 0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        
        except CustomException as e:
            logging.error(e,sys)


# if __name__ == "__main__":
#     data_ingestion=DataIngestion()
#     train_path, test_path = data_ingestion.initiate_data_ingestion()

#     data_transform = DataTransformation()
#     train_arr,test_arr,_ = data_transform.initiate_data_transformation(train_path, test_path)

#     model_trainer = ModelTraining()
#     print(model_trainer.initiate_model_trainer(train_arr,test_arr))



