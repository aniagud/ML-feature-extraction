import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

np.set_printoptions(suppress=True)

# read file
dataset = pd.read_csv("ILPDdata.csv")
dataset_desc = dataset.describe(include="all")

# transform Gender feature to numeric
dataset["Gender"] = dataset["Gender"].replace({"Female": 0, "Male": 1})

X = dataset.iloc[:, :-1].to_numpy()  # array of features arrays
y = dataset.iloc[:, 10].to_numpy()  # array with labels (class)

# dataset_miss = (
#     dataset.isnull().sum()
# )  # One of attributes has NaN values which should be replaced
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 2:10] = imputer.fit_transform(X[:, 2:10])
# X = X[:, 0:1] + X[:, 3:]

# print(dataset_desc)
# print(dataset)
# print(X)
