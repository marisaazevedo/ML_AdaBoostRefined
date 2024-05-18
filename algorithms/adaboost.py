import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:

    def __init__(self, check_outliers=False):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.check_outliers = check_outliers
        self.outliers = []

    def fit(self, X, y, M=100):
        self.G_M = []
        self.alphas = []
        self.training_errors = []
        self.M = M
        self.outliers = self.detect_outliers(X, y) if self.check_outliers else []

        for m in range(0, M):
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            G_m = DecisionTreeClassifier(max_depth=1, max_features=1)
            G_m.fit(X, y, sample_weight=w_i)

            y_pred = G_m.predict(X)

            self.G_M.append(G_m)

            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.M))
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m
        y_pred = (np.sign(weak_preds.T.sum())).astype(int)
        return y_pred

    def detect_outliers(self, X, y):
        outliers = []
        labels = np.unique(y)
        for label in labels:
            data = X[y == label]
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_indices = np.where(z_scores > 3)  # Outlier threshold: 3 std deviations
            outliers.extend(data.index[outlier_indices[0]])
        return outliers

    def compute_error(self, y, y_pred, w_i):
        return sum(w_i * (np.not_equal(y, y_pred)).astype(int)) / sum(w_i)

    def compute_alpha(self, error):
        return 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))

    def update_weights(self, w_i, alpha, y, y_pred):
        mask = np.isin(np.arange(len(y)), self.outliers, invert=True)
        new_w_i = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
        return np.where(mask, new_w_i, w_i)

