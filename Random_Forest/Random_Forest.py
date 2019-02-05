################################################################################
# Utilisation technique Random Forest
################################################################################
################################################################################
# roccad@gmail.com
 
import numpy as np
##import multiprocessing
##from scipy import sparse as sp
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics 
import matplotlib.pyplot as plt
import pickle
from math import sqrt
from math import log2
from matplotlib import ticker, cm 
import time

################################################################################
plt.close()
#start_time = time.time()

# ######################## lecture des donnees #################################

matrice_coulomb = open('matrice_coulomb.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
energie_atomisation = open('energie_atomisation.txt','rb')
energie_atomisation_depickler = pickle.Unpickler(energie_atomisation)
number_of_non_H_atoms = open('number_of_non_H_atoms.txt', 'rb')
number_of_non_H_atoms_depickler = pickle.Unpickler(number_of_non_H_atoms)

X_dataset, Y_dataset= [matrice_coulomb_depickler.load()], [energie_atomisation_depickler.load()]
H_atoms = [number_of_non_H_atoms_depickler.load()]

continuer = True
while continuer == True:
    try:
        X_dataset.append(matrice_coulomb_depickler.load())
        Y_dataset.append(energie_atomisation_depickler.load())
        H_atoms.append(number_of_non_H_atoms_depickler.load())
    except:
        continuer = False    

matrice_coulomb.close()
energie_atomisation.close()
number_of_non_H_atoms.close()
X_dataset = np.asarray(X_dataset)
Y_dataset = np.asarray(Y_dataset)
H_atoms = np.asarray(H_atoms)
# ##################### selection trainnig set et Hold-out set #################

# plt.hist(H_atoms)
# on trie les liste si > ou < a 4 non-H-atoms par molecule
# toute les molecule <4 non H-atoms sont dans le trainning set
inds = H_atoms.argsort()
X_under_4_H = [X_dataset[i] for i in inds if H_atoms[i] < 5]
Y_under_4_H = [Y_dataset[i] for i in inds if H_atoms[i] < 5]
X_above_4_H = [X_dataset[i] for i in inds if H_atoms[i] > 4]
Y_above_4_H = [Y_dataset[i] for i in inds if H_atoms[i] > 4]

# on coupe de maniere aleatoire entre une basse de train (taille 1000) et une base de test
X_train, X_validation, Y_train,Y_validation = model_selection.train_test_split(
    X_above_4_H, Y_above_4_H, train_size= len(X_dataset) - 1000 + len(X_under_4_H),random_state=100) 

# on coupe le base de train en une base de train (900) pour les hyperparametre et une base test (taille 100) pour valider le choix des hyperparametres
X_training_set, X_hold_out_set, Y_training_set,Y_hold_out_set = model_selection.train_test_split(X_train,Y_train,train_size = len(X_train) - 100)  

# on ajoute les molecule <5-non-H-atoms dans la base de train pour avoir taille = 900 
X_training_set = np.concatenate((X_under_4_H,X_training_set),axis=0)
Y_training_set = np.concatenate((Y_under_4_H,Y_training_set),axis =0)   

#print('temps construction dataset= ',time.time() - start_time)

################################################################################
# ###################### entrainement du modele ################################
################################################################################

# #################### choix des hyperparametres ###############################
def RMSE(y_true, y_pred): return sqrt(metrics.mean_squared_error(y_true, y_pred))
def R2(y_true, y_pred): return 1 - (metrics.r2_score(y_true, y_pred))
def MAE(y_true, y_pred): return metrics.mean_absolute_error(y_true, y_pred)




grid_n_estimators = range(1,100,5)
grid_max_depth = range(5,50,5)



# stockage des score pour les differentes mesure
RMSE_SCORE = []
MAE_SCORE = []
R2_SCORE = []
#start_time = time.time()
min_RMSE = 1e20
i = 0
n_estimator_min, max_depth_min = 1,1

for n_estimator in grid_n_estimators:
    for max_depth in grid_max_depth:
        #gamma = 1e-4
        Y_kr_pred =  RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,random_state=2).fit(X_training_set,Y_training_set).predict(X_training_set)     
        RMSE_SCORE.append(RMSE(Y_training_set,Y_kr_pred))
        R2_SCORE.append(R2(Y_training_set,Y_kr_pred))
        MAE_SCORE.append(MAE(Y_training_set,Y_kr_pred))
        if RMSE_SCORE[-1] < min_RMSE: 
            n_estimator_min, max_depth_min = n_estimator, max_depth
            min_RMSE = RMSE_SCORE[-1]
        print(i,'iteration sur',len(grid_max_depth)*len(grid_n_estimators))
        i += 1

print('n_estimator_min=',n_estimator_min,'max_depth_min',max_depth_min,'minimum RMSE sur training',min(RMSE_SCORE))
#print('temps de calcul sur la grille = ',time.time() - start_time)
# calcul erreur sur set taille 100 pour validation
Y_kr_pred = RandomForestRegressor(n_estimators=n_estimator_min, max_depth=max_depth_min,
                                random_state=2).fit(X_training_set,Y_training_set).predict(X_training_set)     
print('erreur sur set validation de taille 100',RMSE(Y_training_set,Y_kr_pred))
#print('temps recherche hyperparametres',time.time() - start_time)


# remplissage sur la grille a partir des scores calcules


X_mesh, Y_mesh = np.meshgrid(grid_n_estimators, grid_max_depth)    
Z_mesh_RMSE = np.zeros(X_mesh.shape)
k = 0
for i in range(Z_mesh_RMSE.shape[0]):
    for j in range(Z_mesh_RMSE.shape[1]):
        Z_mesh_RMSE[i][j] = RMSE_SCORE[k]
        k += 1 
#print('temps remplissage grille = ',time.time() - start_time)

fig, ax = plt.subplots()  

#ax.set_ylim(log2(min(alpha_grid))-200,log2(max(alpha_grid))+200)
#ax.set_xlim(log2(min(gamma_grid))-200,log2(max(gamma_grid))+200)

cp = ax.contourf(X_mesh, Y_mesh, Z_mesh_RMSE)
ax.contour(cp)
ax.clabel(cp, inline=True, fontsize=10)    
ax.grid(c='k', ls='-', alpha=0.7)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
fig.colorbar(cp)
plt.show()   

# affichage p color

fig1, ax1 = plt.subplots()  
cs = ax1.pcolor(X_mesh, Y_mesh, Z_mesh_RMSE, edgecolors='k', linewidths=1)
fig.colorbar(cs)
plt.show()
# ######################## performance modele ##################################

kr_final = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,
                                random_state=2)
kr_final.fit(np.concatenate((X_hold_out_set,X_training_set)),

             np.concatenate((Y_hold_out_set,Y_training_set)))     
Y_kr_pred_final = kr_final.predict(X_validation)
print('score final RMSE =',RMSE(Y_validation,Y_kr_pred_final))
#print('temps execution = ',time.time() - start_time)

matrice_RMSE= open("Resultat_RMSE_RandomForest.txt","wb")
matrice_RMSE_pickler = pickle.Pickler(matrice_RMSE)
for i in range(len(RMSE_SCORE)):
    matrice_RMSE_pickler.dump(RMSE_SCORE[i])
matrice_RMSE.close()
matrice_R2 = open("Resultat_R2_RBF_RandomForest.txt","wb")
matrice_R2_pickler = pickle.Pickler(matrice_R2)
for i in range(len(R2_SCORE)):
    matrice_R2_pickler.dump(R2_SCORE[i])
matrice_R2.close()
matrice_MAE = open("Resultat_MAE_RBF_RandomForest.txt","wb")
matrice_MAE_pickler = pickle.Pickler(matrice_MAE)
for i in range(len(MAE_SCORE)):
    matrice_MAE_pickler.dump(MAE_SCORE[i])
matrice_MAE.close()
