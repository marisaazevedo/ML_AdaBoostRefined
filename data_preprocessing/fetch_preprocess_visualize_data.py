import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def encode_categorical_columns(X, y):
    X_encoded = X.copy()
    y_encoded = y.copy() if isinstance(y, pd.Series) else y[:]

    le_X = LabelEncoder()
    for column in X_encoded.columns:
        if X_encoded[column].dtype == 'object' or X_encoded[column].dtype.name == 'category':
            X_encoded[column] = le_X.fit_transform(X_encoded[column])

    if isinstance(y_encoded, pd.Series) and (y_encoded.dtype == 'object' or y_encoded.dtype.name == 'category'):
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y_encoded)

    return X_encoded, y_encoded


def load_dataset(dataset_id):

    dataset = fetch_openml(data_id=dataset_id)
    X1, y1 = dataset.data, dataset.target
    X, y = encode_categorical_columns(X1, y1)
    return X, y


def apply_pca_and_visualize(X, y=None, name=None, n_components=2):
    # Select numeric features
    features = X.select_dtypes(include=[np.number])

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Plot settings
    plt.figure(figsize=(10, 8), facecolor='white')
    plt.style.use('default')

    # Scatter plot based on y labels
    if y is not None and not y.isna().all():
        unique_labels = np.unique(y)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))
        for color, label in zip(colors, unique_labels):
            indices = y == label
            plt.scatter(principal_df.loc[indices, 'PC1'], principal_df.loc[indices, 'PC2'], alpha=0.7,
                        color=color)  # removed label
    else:
        plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.7)

    # Axes and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(name if name else 'PCA Result')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.show()

    return principal_df


#%%
