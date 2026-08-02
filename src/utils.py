import os
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score

def save_object(file_path,obj_pkl):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj_pkl,file_obj)

    except Exception as e:
        raise CustomException(e, sys)





def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        model_report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            grid_search = GridSearchCV(model,para,cv=3)
            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            #model.fit(X_train, y_train)


            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score
            
        return model_report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    