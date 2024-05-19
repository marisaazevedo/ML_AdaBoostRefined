import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, check_outliers=False, outlier_threshold=3):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.check_outliers = check_outliers
        self.outlier_threshold = outlier_threshold
        self.outliers = None  # To store the outliers

    def fit(self, X, y, M=40):
        self.G_M = []
        self.alphas = []
        self.training_errors = []
        self.M = M

        # Identify outliers if check_outliers is True
        if self.check_outliers:
            self.outliers = self.identify_outliers(X, y)
        else:
            self.outliers = np.zeros(len(y), dtype=bool)

        for m in range(M):
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred, self.outliers)

            G_m = DecisionTreeClassifier(max_depth=1, max_features=1)
            G_m.fit(X, y, sample_weight=w_i)

            y_pred = G_m.predict(X)

            self.G_M.append(G_m)

            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.M))
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m
        y_pred = (np.sign(weak_preds.T.sum())).astype(int)
        return y_pred

    def identify_outliers(self, X, y):
        outliers = np.zeros(len(y), dtype=bool)
        for label in np.unique(y):
            label_indices = np.where(y == label)[0]
            X_label = X.iloc[label_indices]
            z_scores = np.abs((X_label - X_label.mean()) / X_label.std())
            label_outliers = (z_scores > self.outlier_threshold).any(axis=1)
            outliers[label_indices] = outliers[label_indices] | label_outliers
        return outliers

    def get_number_of_outliers(self):
        if self.outliers is None:
            return 0
        return np.sum(self.outliers)


# AUXILIAR FUNCTIONS TO THE ADABOOST CLASS
def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)


def compute_alpha(error):
    return 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))


def update_weights(w_i, alpha, y, y_pred, outliers):
    # Do not update weights for outliers
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred) & ~outliers).astype(int))
