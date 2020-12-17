"""
Podstawy Obliczeń neuronowych
"""

from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
from collections import Counter


E_VARIANCE = 0.9
RANDOM_STATE = 1410
accuracy_function = accuracy_score

rskf = RepeatedStratifiedKFold(
    random_state=RANDOM_STATE
)  # 5 folds, 10 repeats are defaults values

# pca = make_pipeline(
#     StandardScaler(), PCA(n_components=0.9, svd_solver="full", random_state=RANDOM_STATE)
# )
pca = PCA(n_components=E_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
kpca = KernelPCA(kernel="rbf")
lda = LinearDiscriminantAnalysis()

knn = KNeighborsClassifier()  # 5 neighbors are default
svc = svm.SVC(kernel="rbf", random_state=RANDOM_STATE)
mlp = MLPClassifier(
    # solver="sgd",
    # alpha=1e-5,
    hidden_layer_sizes=200,
    max_iter=3000,
    random_state=RANDOM_STATE,
)
gnb = GaussianNB()
cart = DecisionTreeClassifier(random_state=RANDOM_STATE)

dim_reduction_methods = [("PCA", pca), ("KPCA", kpca), ("LDA", lda)]
classification_methods = [
    ("KNN", knn),
    ("SVM", svc),
    ("GNB", gnb),
    ("CART", cart),
    ("MLP", mlp),
]

# Eksperyment 1
print('Eksperyment 1 - wielowymiarowosc danych')

# read file
dataset = pd.read_csv("./datasets/dataset.csv", header=None)
# dataset_desc = dataset.describe(include="all")

features = [5000,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
            1600,1800,2000,2250,2500,2750,3000,3500,4000,4500,5000]

y = dataset.iloc[:, -1].to_numpy()  # array with labels (class)

scores = {}
scores1 = {}
for c_method, _ in classification_methods:
    scores1[c_method] = [[] for _ in range(len(dim_reduction_methods))]

for feature in features:

    X = dataset.iloc[:, :feature].to_numpy()  # array of features arrays
    print(feature)
    print(X.shape)

    scores[feature] = scores1

    for c_method_name, c_method in classification_methods:
        print(c_method_name)
        for i, (r_method_name, r_method) in enumerate(dim_reduction_methods):
            for train_index, test_index in rskf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                c = Counter(y_train)
                cy = Counter(y_test)

                r_method.fit(X_train, y_train)

                X_train_transformed = r_method.transform(X_train)
                X_test_transformed = r_method.transform(X_test)

                c_method.fit(X_train_transformed, y_train)
                y_pred = c_method.predict(X_test_transformed)
                scores[feature][c_method_name][i].append(accuracy_function(y_test, y_pred))

        # no features extraction
        scores[feature][c_method_name].append([])
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            c_method.fit(X_train, y_train)
            y_pred = c_method.predict(X_test)
            scores[feature][c_method_name][len(dim_reduction_methods)].append(
                accuracy_function(y_test, y_pred)
            )

        for i in range(0, len(dim_reduction_methods) + 1):
            print(np.mean(scores[feature][c_method_name][i]))




# KNN
# ...
# D:\Projekty\Machine Learning S9\ML-feature-extraction\venv\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# D:\Projekty\Machine Learning S9\ML-feature-extraction\venv\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# 0.7028359192186909
# 0.0
# 0.5782658716699065
# 0.7494570845752054
# SVM


# Eksperyment 2
print('Eksperyment 2 - dane niezbalansowane')

accuracy_function = balanced_accuracy_score

# read files
imbalanced1 = pd.read_csv("./datasets/imbalanced1.csv", header=None)
imbalanced2 = pd.read_csv("./datasets/imbalanced2.csv", header=None)
imbalanced3 = pd.read_csv("./datasets/imbalanced3.csv", header=None)
imbalanced4 = pd.read_csv("./datasets/imbalanced4.csv", header=None)

imbalanced_datasets = [('imbalanced1', imbalanced1),
                       ('imbalanced2', imbalanced2),
                       ('imbalanced3', imbalanced3),
                       ('imbalanced4', imbalanced4)]

scoresIMB = {}
scores2 = {}
for c_method, _ in classification_methods:
    scores2[c_method] = [[] for _ in range(len(dim_reduction_methods))]

for data_name, dataset in imbalanced_datasets:

    X = dataset.iloc[:, :-1].to_numpy()  # array of features arrays
    y = dataset.iloc[:, -1].to_numpy()  # array with labels (class)
    print(data_name)
    print(X.shape)

    scoresIMB[data_name] = scores2

    for c_method_name, c_method in classification_methods:
        print(c_method_name)
        for i, (r_method_name, r_method) in enumerate(dim_reduction_methods):
            for train_index, test_index in rskf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                r_method.fit(X_train, y_train)

                X_train_transformed = r_method.transform(X_train)
                X_test_transformed = r_method.transform(X_test)

                c_method.fit(X_train_transformed, y_train)
                y_pred = c_method.predict(X_test_transformed)
                scoresIMB[data_name][c_method_name][i].append(accuracy_function(y_test, y_pred))

        # no features extraction
        scoresIMB[data_name][c_method_name].append([])
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            c_method.fit(X_train, y_train)
            y_pred = c_method.predict(X_test)
            scoresIMB[data_name][c_method_name][len(dim_reduction_methods)].append(
                accuracy_function(y_test, y_pred)
            )

        for i in range(0, len(dim_reduction_methods) + 1):
            print(np.mean(scoresIMB[data_name][c_method_name][i]))


# Eksperyment 3
print('Eksperyment 3 - przypadek 3-klasowy')

# read files
triple1 = pd.read_csv("./datasets/triple1.csv", header=None)
triple2 = pd.read_csv("./datasets/triple2.csv", header=None)
triple3 = pd.read_csv("./datasets/triple3.csv", header=None)
triple4 = pd.read_csv("./datasets/triple4.csv", header=None)
triple5 = pd.read_csv("./datasets/triple5.csv", header=None)

triple_datasets = [('triple1', triple1),
                   ('triple2', triple2),
                   ('triple3', triple3),
                   ('triple4', triple4),
                   ('triple5', triple5)]

scoresTR = {}
scores3 = {}
for c_method, _ in classification_methods:
    scores3[c_method] = [[] for _ in range(len(dim_reduction_methods))]

for data_name, dataset in triple_datasets:

    X = dataset.iloc[:, :-1].to_numpy()  # array of features arrays
    y = dataset.iloc[:, -1].to_numpy()  # array with labels (class)
    print(data_name)
    print(X.shape)

    scoresTR[data_name] = scores3

    for c_method_name, c_method in classification_methods:
        print(c_method_name)
        for i, (r_method_name, r_method) in enumerate(dim_reduction_methods):
            for train_index, test_index in rskf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                r_method.fit(X_train, y_train)

                X_train_transformed = r_method.transform(X_train)
                X_test_transformed = r_method.transform(X_test)

                c_method.fit(X_train_transformed, y_train)
                y_pred = c_method.predict(X_test_transformed)
                scoresTR[data_name][c_method_name][i].append(accuracy_function(y_test, y_pred))

        # no features extraction
        scoresTR[data_name][c_method_name].append([])
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            c_method.fit(X_train, y_train)
            y_pred = c_method.predict(X_test)
            scoresTR[data_name][c_method_name][len(dim_reduction_methods)].append(
                accuracy_function(y_test, y_pred)
            )

        for i in range(0, len(dim_reduction_methods) + 1):
            print(np.mean(scoresTR[data_name][c_method_name][i]))
