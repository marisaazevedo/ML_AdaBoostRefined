import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(X, y):
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Convert bool columns to int
    X = X.astype({col: 'int' for col in X.select_dtypes(['bool']).columns})

    # Convert y to numeric, turning non-numeric values into NaN

    # Check if y is of type 'category', if not convert it
    if not isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category')

    # Map categorical labels to numerical values based on their categories
    label_map = {label: idx for idx, label in enumerate(y.cat.categories)}
    y_numeric = y.map(label_map)

    return X, y_numeric


def load_dataset(dataset_id):
    dataset = fetch_openml(data_id=dataset_id)
    X = dataset.data
    y = dataset.target
    X, y = preprocess_data(X, y)
    return X, y


def apply_pca_and_visualize(X, y=None, name=None, n_components=2):
    features = X.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    plt.figure(figsize=(10, 8), facecolor='white')
    plt.style.use('default')

    if y is not None and not y.empty:
        unique_labels = np.unique(y)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))
        for color, label in zip(colors, unique_labels):
            indices = y == label
            plt.scatter(principal_df.loc[indices, 'PC1'], principal_df.loc[indices, 'PC2'], alpha=0.7, color=color, label=label)
        plt.legend()
    else:
        plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.7)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(name)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Adjusted grid for better visibility on white
    plt.show()

    return principal_df