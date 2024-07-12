import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainingConfig:trained_model_file_paths = os.path.join("artifacts/model_train",'model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= RandomForestClassifier()
            models.fit(X_train,y_train)

            y_train_pred = models.predict(X_train)
            y_test_pred = models.predict(X_test)

            # Calculate classification metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            train_precision = precision_score(y_train, y_train_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred, average='weighted')

            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')

            # Store the results
            report = {
                'RandomForestClassifier': {
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'train_accuracy': train_accuracy,
                    'train_f1': train_f1,
                    'train_precision': train_precision,
                    'train_recall': train_recall
                }
            }
            return report
        
        except Exception as e:
            raise CustomException(e,sys)
    

