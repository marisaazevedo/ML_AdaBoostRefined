import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def perform_eda(df, target_column=None):
    print("Statistical Summary:")
    print(df.describe(include='all'))
    print("\n")

    print("Missing Values:")
    print(df.isnull().sum())
    print("\n")

    print("Data Types:")
    print(df.dtypes)
    print("\n")

    if target_column:
        sns.pairplot(df, hue=target_column)
    else:
        sns.pairplot(df)
    plt.show()

    if target_column:
        print("Class Balance:")
        print(df[target_column].value_counts(normalize=True))
        print("\n")

        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=target_column)
        plt.title("Class Distribution")
        plt.show()


# %%
def apply_pca_and_visualize(df, n_components=2):
    """
    Apply PCA on numeric features of the DataFrame and visualize the first two principal components.

    Args:
    df (DataFrame): The DataFrame to process.
    n_components (int, optional): Number of principal components to compute. Defaults to 2.

    Returns:
    DataFrame: DataFrame containing the principal components.
    """
    # Preprocess (selecting numeric features)
    features = df.select_dtypes(include=[np.number])

    # Standardizing the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Applying PCA
    pca = PCA(n_components=n_components)  # Reduce to specified dimensions for visualization
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Visualizing the results
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Dataset')
    plt.grid(True)
    plt.show()

    return principal_df


def label_encode_dataset(df):
    # Create a LabelEncoder object
    le = LabelEncoder()

    # Loop through all columns in the dataframe
    for column in df.columns:
        # Check if the column is of object type (which usually means categorical)
        if df[column].dtype == 'object' or df[column].dtype == 'boolean':            # Apply label encoding to this column
            df[column] = le.fit_transform(df[column])

    return df
