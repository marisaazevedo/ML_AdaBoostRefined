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
    def __init__(self, estimator, n_estimators, learning_rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []

    def fit(self, X, y, estimator=None):
        n_samples, n_features = X.shape
        estimators = ['DecisionTree', 'SVM', 'KNN', 'LogisticRegression', 'Perceptron', 'NaiveBayes']

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
            model = self.estimator
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
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return np.array(scores).mean()

'''
# Criar uma instância do seu modelo AdaBoost
ada_boost = AdaBoost(estimator=DecisionTreeClassifier(), n_estimators=50)

# Calcular a pontuação de validação cruzada k-fold
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
score = ada_boost.cross_val_score(X, y, n_splits=5, random_state=42)

print("Pontuação de validação cruzada:", score)
'''