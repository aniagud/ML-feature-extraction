from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


X, y = datasets.make_classification(n_samples=1000,
                                    n_features=5000,
                                    n_classes=2,
                                    # weights=[0.55, 0.45],
                                    n_informative=4500,
                                    n_repeated=0,
                                    n_redundant=500,
                                    flip_y=.05,
                                    random_state=1410,
                                    n_clusters_per_class=2
                                    )
print(X.shape, y.shape)
# plt.figure(figsize=(5, 2.5))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
# plt.xlabel("$x^1$")
# plt.ylabel("$x^2$")
# plt.tight_layout()
# plt.show()
# plt.savefig('data.png')

dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)
np.savetxt("dataset.csv",
           dataset,
           delimiter=",",
           fmt=["%.5f" for i in range(X.shape[1])] + ["%i"]
           )


# IMBALANCED DATA
for i in range(1, 5):
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=1000,
                                        n_classes=2,
                                        weights=[1-i/10, i/10],
                                        n_informative=100, n_repeated=100, n_redundant=100,
                                        flip_y=.05, random_state=1410, n_clusters_per_class=2)

    imbalanced = np.concatenate((X, y[:, np.newaxis]), axis=1)
    name = "imbalanced" + str(i) + ".csv"
    np.savetxt(name,
               imbalanced,
               delimiter=",",
               fmt=["%.5f" for i in range(X.shape[1])] + ["%i"]
               )


# 3 CLASSES - BALANCED AND IMBALANCED DATA
for i in range(1, 6):
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=1000,
                                        n_classes=3,
                                        weights=[1-i/10, i/10],
                                        n_informative=100, n_repeated=100, n_redundant=100,
                                        flip_y=.05, random_state=1410, n_clusters_per_class=2)

    triple = np.concatenate((X, y[:, np.newaxis]), axis=1)
    name = "triple" + str(i) + ".csv"
    np.savetxt(name,
               triple,
               delimiter=",",
               fmt=["%.5f" for i in range(X.shape[1])] + ["%i"]
               )
