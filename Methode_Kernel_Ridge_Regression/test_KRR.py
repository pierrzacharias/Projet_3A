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


# ##############################################################################
# generation dataset pour regression avec 2 parametre pour visualisation
X_dataset, y_dataset = datasets.make_regression(n_samples=1000, n_features=1, n_targets=1, bias=0.0, tail_strength=10, noise = 30, shuffle=True)
#plt.figure()
#plt.scatter(X_dataset[:200], y_dataset[:200],c = 'r', s = 10)
#plt.title("dataset")
#plt.show()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset,y_dataset,test_size=.5)

kr = KernelRidge(kernel='rbf', gamma=0.9)
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

print("score KRR %.3f" % kr.score(X_test, y_test) )


################################################################################
#                        optimisation des parametres
################################################################################

# recherche des paramètre optimaux 
# alpha = (2*C)^-1, C = 1/2*alpha
kr_opti = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.5), cv=5,
                  param_grid={"alpha": [np.logspace(0, 1, 50)],
                              "gamma": np.logspace(-10, 0.1,10)},
                  return_train_score = True)
# can use cv for cross-valisation 

kr_opti.fit(X_train, y_train)

#print(kr_opti.cv_results_.key())

print("score KRR OTIMISEE %.3f" % kr_opti.score(X_test, y_test) )

y_kr_opti = kr_opti.predict(X_plot)
plt.plot(X_plot, y_kr_opti,color='b', marker='d', linestyle='-',label='KKR OPTI regression')
plt.show()


# gamma trop petit : sous-apprentissage car inversement proportionnel a la variance gaussienne
# si C augmente plus de vecteur support
#plt.figure()
#alpha_scope = np.arange(0.01,0.5,0.1)
#gamma_scope = np.arange(0,0.05,0.0001)
#for alpha in alpha_scope:
    #SCORE = []
    #for gamma in gamma_scope:
        
        #kr_estim = KernelRidge(kernel='rbf', alpha = alpha, gamma = gamma)
        #kr_estim.fit(X_train,y_train)
        #SCORE.append(kr_estim.score(X_test, y_test))
    #plt.plot(gamma_scope,SCORE, label='alpha =%.3f'%alpha)
#plt.title("optimisation")
#plt.legend()
#plt.show()