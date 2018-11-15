################################################################################
# <<<<<<<<<<<<<<<<< Essai implementation methode KRR >>>>>>>>>>>>>>>>>>>>>>>>>>>
################################################################################

################################################################################
import numpy as np
from sklearn import datasets,model_selection
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
################################################################################
rng = np.random.RandomState(0)

# ##############################################################################
# generation dataset pour regression avec 2 parametre pour visualisation
X_dataset, y_dataset = datasets.make_regression(n_samples=1000, n_features=1, n_targets=1, bias=0.0, tail_strength=10, noise = 0.1, shuffle=True)
#plt.figure()
#plt.scatter(X_dataset[:200], y_dataset[:200],c = 'r', s = 10)
#plt.title("dataset")
#plt.show()

# #############################################################################
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset,y_dataset,test_size=.5)

kr = KernelRidge(kernel='rbf', gamma=0.1)
kr.fit(X_train, y_train)

X_plot =  np.linspace(float(min(X_dataset)), float(max(X_dataset)),50)[:, None]   
y_kr = kr.predict(X_plot)

# coefficient directeurs
sv_ind = kr.dual_coef_
plt.figure()
plt.scatter(X_dataset[:200], y_dataset[:200], c='r', s = 30,label='dataset')

plt.plot(X_plot, y_kr,color='green', marker='o', linestyle='-',label='KKR regression')
plt.title("KKR regression")
plt.legend()
plt.show()

print("score KRR %.3f" % kr.score(X_test, y_test) )