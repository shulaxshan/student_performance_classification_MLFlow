import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import lightgbm as lgb
from src.utils import confustion_matrix

from src.exception import CustomException
from src.logger import logging

from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

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

            # models= RandomForestClassifier(n_estimators=200,min_samples_split=2,max_depth=5)
            # models.fit(X_train,y_train)

            # y_train_pred = models.predict(X_train)
            # y_test_pred = models.predict(X_test)

            # # Calculate classification metrics
            # train_accuracy = accuracy_score(y_train, y_train_pred)
            # test_accuracy = accuracy_score(y_test, y_test_pred)

            # train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            # test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # train_precision = precision_score(y_train, y_train_pred, average='weighted')
            # test_precision = precision_score(y_test, y_test_pred, average='weighted')

            # train_recall = recall_score(y_train, y_train_pred, average='weighted')
            # test_recall = recall_score(y_test, y_test_pred, average='weighted')

            # # Store the results
            # report = {
            #     'RandomForestClassifier': {
            #         'test_accuracy': test_accuracy,
            #         'test_f1': test_f1,
            #         'test_precision': test_precision,
            #         'test_recall': test_recall,
            #         'train_accuracy': train_accuracy,
            #         'train_f1': train_f1,
            #         'train_precision': train_precision,
            #         'train_recall': train_recall
            #     }
            # }
            #return report
            n_estimators =  int(sys.argv[1]) if len(sys.argv)>1  else 200
            criterion = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ["gini", "entropy", "log_loss"] else "gini"
            min_samples_split = int(sys.argv[3]) if len(sys.argv)>3  else 2

            
            with mlflow.start_run():
                models= RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,min_samples_split=min_samples_split)
                models.fit(X_train, y_train)
                prediction = models.predict(X_test)

                (testing_auc, testing_precision, testing_recall, testing_f1) = confustion_matrix(y_test, prediction)

                print("  Accuracy: %s" % testing_auc)
                print("  Precision: %s" % testing_precision)
                print("  Recall: %s" % testing_recall)
                print("  F1-score: %s" % testing_f1)
                print("  Selected n_estimators: %s" % n_estimators)
                print("  Selected criterion: %s" % criterion)
                print("  Selected min_samples_split: %s" % min_samples_split)

                clf_params = models.get_params()
                mlflow.log_params(clf_params)
                mlflow.log_metric("Testing_auc",testing_auc)
                mlflow.log_metric("Testing_precision",testing_precision)
                mlflow.log_metric("Testing_recall",testing_recall)
            

                signature = infer_signature(X_test, prediction)
                #model_info = mlflow.sklearn.log_model( sk_model=models, artifact_path="model", signature=signature)

  

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(models, "model", registered_model_name="RandomForestClassifier", signature=signature)
                else:
                    mlflow.sklearn.log_model(models, "model", signature=signature)

        
        except Exception as e:
            raise CustomException(e,sys)
    

