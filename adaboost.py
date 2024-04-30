import numpy as np


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionStump()
            model.fit(X, y, sample_weight=weights)
            predictions = model.predict(X)

            error = np.sum(weights * (predictions != y))

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for model, alpha in zip(self.models, self.alphas):
            predictions += alpha * model.predict(X)

        return np.sign(predictions)


class DecisionStump:
    def __init__(self):
        self.threshold = None
        self.feature_index = None
        self.prediction = None

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_index in range(n_features):
            feature_values = np.unique(X[:, feature_index])
            for threshold in feature_values:
                predictions = np.ones(n_samples)
                predictions[X[:, feature_index] < threshold] = -1
                error = np.sum(sample_weight[predictions != y])

                if error < min_error:
                    min_error = error
                    self.threshold = threshold
                    self.feature_index = feature_index

                    self.prediction = 1 if np.mean(y[predictions == 1]) > np.mean(y[predictions == -1]) else -1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        predictions[X[:, self.feature_index] < self.threshold] = -1

        return predictions
