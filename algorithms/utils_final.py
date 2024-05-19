import openml
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder


def obter_dataset(id):
    dataset = openml.datasets.get_dataset(id, download_data=True, download_qualities=True,
                                          download_features_meta_data=True)

    X, y, _, attrs = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

    df = pd.DataFrame(X, columns=attrs)
    df['target'] = y
    # converter:
    #   0 -> -1
    #   1 -> 1
    df['target'] = 2 * y - 1
    # erase rows with NaN values
    df = df.dropna(how='any', axis=0)

    csv_path = f'{id}.csv'
    df.to_csv(csv_path, index=False)

    return df


def perform_eda(df, target_column=None):
    print("Basic Information:")
    print(df.info())
    print("\n")

    print("Statistical Summary:")
    print(df.describe(include='all'))
    print("\n")

    print("Missing Values:")
    print(df.isnull().sum())
    print("\n")

    print("First 5 Rows:")
    print(df.head())
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


def run_cv(X, y, algs, nfolds=10, means_only=False):
    results = {}
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=1111)
    for algo_name, algo in algs:
        results[algo_name] = []
        sum_fold = 0
        number_of_outliers = 0
        print("\n")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)
            results[algo_name].append(accuracy_score(y_test, y_pred))
            sum_fold += results[algo_name][-1]
            # print(f"Fold {fold} for {algo_name}: {results[algo_name][-1]}")
            if algo_name == "AdaBoost Check Outliers 3":
                number_of_outliers += algo.get_number_of_outliers()

        print(f"Number of outliers identified: {number_of_outliers / nfolds}")
        print(f"Mean for {algo_name}: {sum_fold / nfolds}")
    results_df = pd.DataFrame.from_dict(results)
    if not means_only:
        return results_df
    else:
        results_means = {}
        for algo_name, algo in algs:
            results_means[algo_name] = [np.mean(results[algo_name])]
        return pd.DataFrame.from_dict(results_means)


def plot_cv(results_cv, metric='Accuracy', title="Cross-validation results for multiple algorithms in a single task"):
    fig, ax = plt.subplots()
    ax.boxplot(results_cv)
    ax.set_xticklabels(results_cv.columns)
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.show()


# Análise dos outliers de cada dataset
'''
O valor que você deve colocar no limite depende do que você considera um outlier. No código que você forneceu, você está usando a regra do escore z para detectar outliers. Especificamente, você está considerando qualquer valor que seja mais de 3 desvios padrão da média como um outlier.  
O valor de 3 é comumente usado na regra do escore z porque corresponde a um nível de confiança de cerca de 99,7% em uma distribuição normal. Isso significa que, em uma distribuição normal, esperamos que cerca de 99,7% dos valores estejam dentro de 3 desvios padrão da média.

'''


def showoutliers(ids, show_Boxplot=True):
    limite = 5

    df_1 = dataframes[id][dataframes[id]['target'] == 1]
    df_1 = df_1.drop(columns=['target'], axis=1)
    df_not_1 = dataframes[id][dataframes[id]['target'] == -1]
    df_not_1 = df_not_1.drop(columns=['target'], axis=1)

    outliers_1 = set()
    outliers_not_1 = set()

    for coluna in df_1.columns:

        data_plot_1 = df_1[coluna]
        data_plot_not_1 = df_not_1[coluna]

        if show_Boxplot:
            print(f'Outliers em {coluna}:')
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

            # Plot para a classe 1
            axs[0].boxplot(df_1[f'{coluna}'], vert=False)
            axs[0].set_title('Boxplot para df_1')

            # Plot para a classe -1
            axs[1].boxplot(df_not_1[f'{coluna}'], vert=False)
            axs[1].set_title('Boxplot para df_not_1')

            plt.show()

        z_scores_1 = np.abs((data_plot_1 - data_plot_1.mean()) / data_plot_1.std())
        outliers_1.update(data_plot_1[z_scores_1 > limite].index)
        #print(f"outliers 1: {outliers_1}")

        z_scores_not_1 = np.abs((data_plot_not_1 - data_plot_not_1.mean()) / data_plot_not_1.std())
        outliers_not_1.update(data_plot_not_1[z_scores_not_1 > limite].index)
        #print(f"outliers -1: {outliers_not_1}")

    n_outliers = len(outliers_1) + len(outliers_not_1)
    return n_outliers


'''
Quando estamos a analisar os outliers fará sentido separar as classes (ver o boxplot para cada classe separadamente) ou é melhor analisar todas as classes juntas?
A análise de outliers deve ser feita para cada classe separadamente. Se você analisar todas as classes juntas,
 você pode acabar considerando valores que são normais para uma classe como outliers, simplesmente porque eles são incomuns em relação a outra classe.
'''
