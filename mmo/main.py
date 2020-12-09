from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score

import preprocessing

X = preprocessing.X
y = preprocessing.y

E_VARIANCE = 0.9
RANDOM_STATE = 1410
accuracy_function = precision_score

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

scores = {}
for c_method, _ in classification_methods:
    scores[c_method] = [[] for _ in range(len(dim_reduction_methods))]

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
            scores[c_method_name][i].append(accuracy_function(y_test, y_pred))

    # no features extraction
    scores[c_method_name].append([])
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        c_method.fit(X_train, y_train)
        y_pred = c_method.predict(X_test)
        scores[c_method_name][len(dim_reduction_methods)].append(
            accuracy_function(y_test, y_pred)
        )

    for i in range(0, len(dim_reduction_methods) + 1):
        print(np.mean(scores[c_method_name][i]))
