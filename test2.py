from sklearn import datasets, model_selection, preprocessing, model_selection, svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
X_dataset, y_dataset = datasets.make_regression(n_samples=1000, n_features=2, n_targets=1, bias=0.0, tail_strength=10, noise=4, shuffle=True)
plt.figure()
plt.scatter(X_dataset[:, 0], X_dataset[:, 1], marker='o', c=y_dataset,
            s=25, edgecolor='k')
plt.title('dataset')
plt.show()