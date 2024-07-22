from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        train_data_path = 'artifacts/data_ingestion/train.csv'
        test_data_path = 'artifacts/data_ingestion/test.csv'
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx===========x")
    except CustomException as e:
        logging.error(e,sys)
