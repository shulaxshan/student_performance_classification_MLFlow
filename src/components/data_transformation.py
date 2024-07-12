import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_tranformer_object(self):
        try:
            numerical_columns = ["Student_Age","Weekly_Study_Hours"]
            one_hot_categorical_columns = ["High_School_Type","Scholarship"]
            
            num_pipline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())]
            )

            cat_pipline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder()),
                         ("scaler", StandardScaler(with_mean=False))]
            )


            logging.info(f"One Hot Categorical columns: {one_hot_categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            
            preprocessor = ColumnTransformer(
                transformers = [("num", num_pipline, numerical_columns),
                                ("cat", cat_pipline, one_hot_categorical_columns)]
            )

            return preprocessor
        
        except CustomException as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Initiating data transformation")
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f"Train data shape: {train_data.shape}")
            logging.info(f"Test data shape: {test_data.shape}")

            logging.info("Obtaining preprocessing objects")
            preprocessor_obj = self.get_data_tranformer_object()

            target_column_name="Grade"

            input_feature_train_df = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            logging.info(f"Input training df features: {input_feature_train_df.columns}")
            logging.info(f"Input testing df features: {input_feature_test_df.columns}")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        

        except Exception as e:
            raise CustomException(e,sys)
