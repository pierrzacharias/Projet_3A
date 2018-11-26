################################################################################
# <<Dans ce sript nous entrainons le modele Kernel Ridge sur le jeu de donnee>>>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
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

X_dataset, Y_dataset = np.array([matrice_coulomb_depickler.load()]), np.array([energie_atomisation_depickler.load()])

continuer = True
i = 0
while continuer == True:
    try:
        X_dataset = np.insert(X_dataset,i,matrice_coulomb_depickler.load(), axis=0)
        Y_dataset = np.append(Y_dataset,energie_atomisation_depickler.load())
        i += 1
    except:
        continuer = False    
# print(X_dataset[:10])
matrice_coulomb.close()
energie_atomisation.close()

# autre methode data_array = np.loadtxt("matrice_coulomb.txt")
# ###################### entrainement du modèle ################################

# on separe la base en une base de test et une base d'apprentissage
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset, Y_dataset, test_size=.1)

# recherche du parametre optimal 
# alpha = (2*C)^-1, C = 1/2*alpha
kr_opti = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [np.logspace(0, 1, 50)],
                              "gamma": np.logspace(-10, 0.1,10)}, verbose=True,
                  return_train_score = True)
 
kr_opti.fit(X_train, y_train)

print("score KRR OTIMISEE %.3f" % kr_opti.score(X_test, y_test) )
print("paramètres optimisés", kr_opti.best_params_)

# X_plot =  np.linspace(float(min(X_dataset)), float(max(X_dataset)),50)[:, None] 
#y_kr_opti = kr_opti.predict(X_plot)
#plt.plot(X_plot, y_kr_opti,color='b', marker='d', linestyle='-',label='KKR OPTI regression')
#plt.show()





