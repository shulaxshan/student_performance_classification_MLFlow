from src.exception import CustomException
from src.logger import logging
import sys
from src.components.model_trainer import ModelTraining
import numpy as np


STAGE_NAME = "Model training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        train_arr_data_path = 'artifacts/data_transformation/preprocessor_trained_data.npy'
        test_arr_data_path = 'artifacts/data_transformation/preprocessor_test_data.npy'

        # Load the numpy arrays
        train_arr = np.load(train_arr_data_path)
        test_arr = np.load(test_arr_data_path)
        print(train_arr)
        
        logging.info(f"Model training started...")

        model_trainer = ModelTraining() 
        model_trainer.initiate_model_trainer(train_arr,test_arr)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx===========x")
    except CustomException as e:
        logging.error(e,sys)


