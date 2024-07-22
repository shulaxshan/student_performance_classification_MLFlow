import sys
from src.exception import CustomException
from src.logger import logging
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_03_model_tainer import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logging.info(f">>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===========x")
except CustomException as e:
        logging.error(e,sys)


STAGE_NAME = "Data Transformation stage"
try:
    logging.info(f"****************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed___ <<<<<<<<<\n\nx===========x")
except CustomException as e:
        logging.error(e,sys)


STAGE_NAME = "Model training stage"
try:
    logging.info(f"****************")
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    model_train = ModelTrainingPipeline()
    model_train.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx===========x")
except CustomException as e:
        logging.error(e,sys)