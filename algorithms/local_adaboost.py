import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class AdaBoost:
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            # Clone the base estimator
            model = self.base_estimator()
            # Train the model
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Calculate the error
            err = np.sum(w * (y_pred != y)) / np.sum(w)

            # Calculate the alpha value (amount of say of the model)
            alpha = self.learning_rate * np.log((1 - err) / (err + 1e-10)) / 2

            # Update weights
            w *= np.exp(-alpha * y * y_pred)

            # Normalize weights
            w /= np.sum(w)

            # Save model and alpha
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Aggregate predictions from each model
        preds = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            preds += alpha * model.predict(X)
        return np.sign(preds)

