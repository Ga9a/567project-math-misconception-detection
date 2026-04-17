import numpy as np
from scipy import sparse

X_train = sparse.load_npz("features/X_train.npz")
X_val = sparse.load_npz("features/X_val.npz")
X_test = sparse.load_npz("features/X_test.npz")

y_train = np.load("features/y_train.npy")
y_val = np.load("features/y_val.npy")

print(X_train.shape,X_val.shape,X_test.shape)