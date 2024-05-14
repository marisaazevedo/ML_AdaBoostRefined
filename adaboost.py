import numpy as np
from sklearn.svm import SVC


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, x, y):
        n_samples, n_features = x.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = SVC(kernel='linear', probability=True, class_weight='balanced')
            model.fit(x, y, sample_weight=weights)
            predictions = model.predict(x)

            error = np.sum(weights * (predictions != y))
            # Adding a small constant to prevent division by zero
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, x):
        n_samples = x.shape[0]
        predictions = np.zeros(n_samples)

        for model, alpha in zip(self.models, self.alphas):
            predictions += alpha * model.predict(x)

        return np.sign(predictions)
