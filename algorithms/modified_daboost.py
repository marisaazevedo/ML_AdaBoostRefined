import numpy as np
from sklearn.tree import DecisionTreeClassifier


class ModifiedAdaboost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.model_weights = []

    def fit(self, x, y, outliers):
        n_samples = x.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(x, y, sample_weight=weights)
            pred = model.predict(x)

            misclassified = pred != y
            correct = ~misclassified
            weighted_error = np.dot(weights, misclassified) / np.sum(weights)

            if weighted_error == 0:
                break

            beta = weighted_error / (1 - weighted_error)

            weights[misclassified] *= np.where(outliers[misclassified], 1.0, beta)
            weights[correct] *= np.where(outliers[correct], 1.0, 1 / beta)

            weights /= np.sum(weights)

            self.models.append(model)
            self.model_weights.append(np.log(1 / beta))

    def predict(self, x):
        model_pred = np.array([model.predict(x) for model in self.models])
        final_pred = np.sign(np.dot(self.model_weights, model_pred - 0.5))
        return (final_pred >= 0).astype(int)
