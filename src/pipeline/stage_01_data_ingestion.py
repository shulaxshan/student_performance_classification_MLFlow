from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx===========x")
    except CustomException as e:
        logging.error(e,sys)