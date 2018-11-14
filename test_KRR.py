################################################################################
# <<<<<<<<<<<<<<<<< Essai implementation methode KRR >>>>>>>>>>>>>>>>>>>>>>>>>>>
################################################################################

################################################################################
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
################################################################################
rng = np.random.RandomState(0)

# ##############################################################################
# generation dataset pour regression avec 2 parametre pour visualisation
X_dataset, y_dataset = datasets.make_regression(n_samples=1000, n_features=2, n_targets=1, bias=0.0, tail_strength=10, noise=4, shuffle=True)
plt.figure()
plt.scatter(X_dataset[:, 0], X_dataset[:, 1], marker='o', c=y_dataset,
            s=25, edgecolor='k')
plt.title('dataset')
plt.show()

# #############################################################################
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset,y_dataset,test_size=.5)

kr = KernelRidge(kernel='rbf', gamma=0.1)



kr.fit(X_train, y_train)



y_kr = kr.predict(X_dataset)

