import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ModelTrainer():
    """ Creates trains and stores machine learning models"""
    def __init__(self, name: str):
        self.name = name
        if name == 'SVC':
            self.model = SVC(max_iter=1000)
        elif name == 'LOGISTIC REGRESSION': 
            self.model = LogisticRegression(max_iter=10000)
        elif name == 'RANDOM FOREST CLASSIFIER':
            self.model = RandomForestClassifier(max_depth=200)
        elif name == 'MLP CLASSIFIER': 
            self.model = MLPClassifier(max_iter=1000)
        else:
            raise ValueError('Not a valid model name, select between "SVC", "LOGISTIC REGRESSION", "RANDOM FOREST CLASSIFIER", and "MLP CLASSIFIER".') 
        self.scaler = StandardScaler()

    def train_model(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """Creates a model and saves it in a file for later use"""

        x_train_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_train_scaled, y_train.values.ravel())

        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.model, 'model.pkl')