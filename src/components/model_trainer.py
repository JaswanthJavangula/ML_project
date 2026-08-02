import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,  AdaBoostRegressor

from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                    train_array[:, :-1], train_array[:, -1], 
                    test_array[:, :-1], test_array[:,-1]
            )

            model = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
                "Decision Tree": {
                     "splitter":['best', 'random'],
                     "max_features":['sqrt', 'log2'],
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2']
                    },
                "Gradient Boosting": {
                     "loss": ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "max_features": ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }



            #Evaluate model will be created in utils.py file and we will call that function here to get the best model
            model_report:dict =evaluate_models(
                    X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= model, param = params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = model[best_model_name]

            if best_model_score < 0.6:
                    raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            save_object(
                    file_path = self.model_trainer_config.trained_model_file_path,
                    obj_pkl = best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
                raise CustomException(e, sys)