import os
import sys
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def confustion_matrix(X_test, X_test_pred):
    testing_auc = accuracy_score(X_test, X_test_pred)
    testing_precision = precision_score(X_test, X_test_pred, average='weighted')
    testing_recall = recall_score(X_test, X_test_pred, average='weighted')
    testing_f1 = f1_score(X_test, X_test_pred, average='weighted')
    return testing_auc, testing_precision, testing_recall, testing_f1