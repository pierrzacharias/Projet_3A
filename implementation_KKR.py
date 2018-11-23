################################################################################
# <<<<<<<<<<<<<<<<<<<<<< use KernelRidge methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
################################################################################

################################################################################
import numpy as np
from sklearn import datasets,model_selection
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pickle

################################################################################


# ######################## lecture des donnees #################################
matrice_coulomb = open('matrice_coulomb.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
energie_atomisation = open('energie_atomisation.txt','rb')
energie_atomisation_depickler = pickle.Unpickler(energie_atomisation)

X_dataset, Y_dataset = [], []
continuer = True
while continuer == True:
    try:
        X_dataset.append( matrice_coulomb_depickler.load() )
        Y_dataset.append( energie_atomisation_depickler.load() )
    except:
        continuer = False    
# print(X_dataset[:10])
matrice_coulomb.close()
energie_atomisation.close()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset,Y_dataset,test_size=.1)
i = 0
continuer = True
while continuer:
    try:
        X_dataset[i]
        i += 1
    except:
        continuer = False
print(i)
i = 0
continuer = True
while continuer:
    try:
        Y_dataset[i]
        i += 1
    except:
        continuer = False
print(i)
# ###################### modele KernelRidge ####################################

# alpha = (2*C)^-1, C = 1/2*alpha
kr_opti = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [np.logspace(0, 1, 50)],
                              "gamma": np.logspace(-10, 0.1,10)}, verbose=True,
                  return_train_score = True)
# can use cv for cross-valisation 

kr_opti.fit(X_train, y_train)

print("score KRR OTIMISEE %.3f" % kr_opti.score(X_test, y_test) )
print("paramètres optimisés", kr_opti.best_params_)
y_kr_opti = kr_opti.predict(X_plot)
plt.plot(X_plot, y_kr_opti,color='b', marker='d', linestyle='-',label='KKR OPTI regression')
plt.show()





