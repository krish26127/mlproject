import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            
            params = {

                "Random Forest": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [1.0, 0.8, 0.6],
                    "max_depth": [3, 4, 5]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "copy_X": [True, False]
                    # Linear Regression has very few hyperparameters
                },

                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]   # 1 = Manhattan, 2 = Euclidean
                },

                "XGBClassifier": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.3, 0.1, 0.05, 0.01],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [1, 0.8, 0.6],
                    "colsample_bytree": [1, 0.8, 0.6]
                },

                "CatBoosting Classifier": {
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [200, 500, 800]
                    # CatBoost handles categorical features automatically
                },

                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [1.0, 0.5, 0.1, 0.01]
                }
            }


            model_report: dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # FIX THIS ALSO — incorrect method name
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
