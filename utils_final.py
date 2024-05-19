import openml
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# configurar o estilo dos gráficos para a cor de fundo ser branca
sns.set_style("white")

# esta função é usada para obter um dataset do OpenML
def obter_dataset(id):
    dataset = openml.datasets.get_dataset(id, download_data=False, download_qualities=False, download_features_meta_data=False)

    X, y, _, attrs = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

    df = pd.DataFrame(X, columns=attrs)
    df['target'] = y
    # converter:
    #   0 -> -1
    #   1 -> 1
    df['target'] = 2 * y - 1
    # erase rows with NaN values
    df = df.dropna(how='any', axis=0)

    return df


# esta função é usada para fazer cross-validation
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


# esta função é usada para identificar os outliers e visualizá-los
def show_outliers(df, show_Boxplot=True):
    limite = 5

    df_1 = df[df['target'] == 1]
    df_1 = df_1.drop(columns=['target'], axis=1)
    df_not_1 = df[df['target'] != 1]
    df_not_1 = df_not_1.drop(columns=['target'], axis=1)

    outliers_1 = set()
    outliers_not_1 = set()

    if len(df_1.columns) > 10:
        lenght = 10
    else:
        lenght = len(df_1.columns)

    for coluna in df_1.columns:
        data_plot_1 = df_1[coluna]
        data_plot_not_1 = df_not_1[coluna]

        if lenght != 0:
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
                lenght -= 1

        z_scores_1 = np.abs((data_plot_1 - data_plot_1.mean()) / data_plot_1.std())
        outliers_1.update(data_plot_1[z_scores_1 > limite].index)
        #print(f"outliers 1: {outliers_1}")

        z_scores_not_1 = np.abs((data_plot_not_1 - data_plot_not_1.mean()) / data_plot_not_1.std())
        outliers_not_1.update(data_plot_not_1[z_scores_not_1 > limite].index)
        #print(f"outliers -1: {outliers_not_1}")

    n_outliers = len(outliers_1) + len(outliers_not_1)
    percentage_outliers = n_outliers / (len(df))
    print(f'Número de outliers: {n_outliers}')
    print(f'Número total de amostras: {len(df)}')
    print(f'Percentagem de outliers: {percentage_outliers * 100:.2f}%')


# esta função é usada para analisar as informações do dataset
def analyze_dataset(df):
    # Prints de informação importante sobre o dataset
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\nColumn Names:", df.columns.tolist())
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nColumns with Missing Values:", df.columns[df.isnull().any()].tolist())
    if len(df.columns[df.isnull().any()]) > 0:
        print("\nNumber of rows with Missing Values:", len(pd.isnull(df).any(1).nonzero()[0].tolist()))
    print("\nSample Rows:\n", df.head())


# esta função é usada para visualizar vários aspectos do dataset
def visualize_dataset(df):
    plt.figure(figsize=(15, 10))

    num_cols = df.select_dtypes(include=np.number).columns

    # Gráfico heatmap para correlação
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    aa = df.cumsum()
    aa.plot()

    # PCA
    df_num = df.select_dtypes(include=np.number).dropna()
    if df_num.empty:
        return # sem colunas numéricas

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)

    # Transformar os dados em componentes principais
    pca = PCA()
    pca_components = pca.fit_transform(df_scaled)

    # Fazer o gráfico dos componentes principais
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], c=pca_components[:, 1], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA - First Two Principal Components')
    sns.despine()
    plt.show()

def plot_cm(model_fit, X_test, y_test, id):
    y_pred = model_fit.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title(f'Confusion Matrix for Dataset {id}')
    plt.show()


def plot_roc_curve(y_true, y_pred, id):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='lightblue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Dataset {id}')
    plt.legend(loc="lower right")
    plt.show()