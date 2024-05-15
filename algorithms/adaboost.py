import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, estimator='default', n_estimators=50, learning_rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []

    def fit(self, X, y, estimator):
        n_samples, n_features = X.shape

        if estimator == 'DecisionTree':
            model = DecisionTreeClassifier(max_depth=1)
        elif estimator == 'SVM':
            model = SVC()
        elif estimator == 'KNN':
            model = KNeighborsClassifier()
        elif estimator == 'LogisticRegression':
            model = LogisticRegression()
        elif estimator == 'Perceptron':
            model = Perceptron()
        elif estimator == 'NaiveBayes':
            model = GaussianNB()
        else:
            # default model
            model = DecisionTreeClassifier()

        model.fit(X, y)
        self.models.append(model)
        self.alphas.append(1.0)
        w = np.full(n_samples, 1 / n_samples)
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Compute the error
            err = np.sum(w * (y_pred != y)) / np.sum(w)

            # Compute the alpha value
            alpha = self.learning_rate * np.log((1 - err) / (err + 1e-10))

            # Update weights
            w *= np.exp(alpha * (y_pred != y))

            # Normalize weights
            w /= np.sum(w)

            # Save the model and alpha
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Aggregate predictions from each model
        pred = sum(alpha * model.predict(X) for model, alpha in zip(self.models, self.alphas))
        return np.sign(pred)

    def cross_val_score(self, X, y, n_splits=5, random_state=None):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.fit(X_train, y_train, self.estimator)
            y_pred = self.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return np.array(scores).mean()