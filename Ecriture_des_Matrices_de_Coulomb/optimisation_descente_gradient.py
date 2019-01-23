################################################################################
#   Dans ce sript nous entrainons le modele Kernel Ridge sur le jeu de donnee
#   On optilise avec la methode du gradient
################################################################################
# os.chdir("C:/Users/pierr/Documents/3A/projet/Projet_3A/Methode_Kernel_Ridge_Regression")
################################################################################
import numpy as np
from sklearn import datasets,model_selection
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
start_time = time.time()
# ######################## lecture des donnees #################################

matrice_coulomb = open('matrice_coulomb.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
energie_atomisation = open('energie_atomisation.txt','rb')
energie_atomisation_depickler = pickle.Unpickler(energie_atomisation)
number_of_non_H_atoms = open('number_of_non_H_atoms.txt', 'rb')
number_of_non_H_atoms_depickler = pickle.Unpickler(number_of_non_H_atoms)

X_dataset, Y_dataset= np.array([matrice_coulomb_depickler.load()]), np.array([energie_atomisation_depickler.load()])
H_atoms = np.array([number_of_non_H_atoms_depickler.load()])
    
continuer = True
i = 0
while continuer == True:
    try:
        X_dataset = np.insert(X_dataset,i,matrice_coulomb_depickler.load(), axis=0)
        Y_dataset = np.append(Y_dataset,energie_atomisation_depickler.load())
        H_atoms = np.append(H_atoms,number_of_non_H_atoms_depickler.load())
        i += 1
    except:
        continuer = False  
        
# ##################### selection trainnig set et Hold-out set #################
# plt.hist(H_atoms)
# on trie les liste si > ou < a 4 non-H-atoms par molecule
# ce la permet de mettre toutes le molecule <5 non-H atmos dentrainer le modele
# car il y en a peu comparee aux autres molecules

inds = H_atoms.argsort()
X_under_4_H = [X_dataset[i] for i in inds if H_atoms[i] < 5]
Y_under_4_H = [Y_dataset[i] for i in inds if H_atoms[i] < 5]
X_above_4_H = [X_dataset[i] for i in inds if H_atoms[i] > 4]
Y_above_4_H = [Y_dataset[i] for i in inds if H_atoms[i] > 4]

# on coupe de maniere aleatoire entre une basse de train et une base de test
X_train, X_validation, Y_train,Y_validation = model_selection.train_test_split(
X_above_4_H, Y_above_4_H, train_size= 1000- len(X_under_4_H), random_state=200) 

# on coupe le base de t rain en une base de train generale et un set de test pour choisir
# les hyperparametres

X_training_set, X_hold_out_set, Y_training_set,Y_hold_out_set = model_selection.train_test_split(
X_train,Y_train,test_size= 100,train_size = 900 - len(X_under_4_H),random_state=100)  

# on ajoute les molecule <5-non-H-atoms dans la base de train 

X_training_set = np.concatenate((X_under_4_H,X_training_set),axis=0)
Y_training_set = np.concatenate((Y_under_4_H,Y_training_set))         

################################################################################
# ###################### entrainement du modele ################################
################################################################################

# #################### choix des hyperparametres ###############################

# on entraine sur trainning_set et on predit sur hold_out_set
# recherche du parametre optimal 
# alpha = (2*C)^-1, C = 1/2*alpha
# dans article sigma de 5 a 18 --> gamma = 1/sigma**2 gamma in [.003,0.04]
# lambda de -40 a -5
# print(X_dataset[:10])
# gamma = 1/sigma**2 
param_grid={"alpha": [np.linspace(-30, -5, 20)]}

def RMSE(y_true, y_pred): return sqrt(metrics.mean_squared_error(y_true, y_pred))
def R2(y_true, y_pred): return 1 - (metrics.r2_score(y_true, y_pred))
def MAE(y_true, y_pred): return metrics.mean_absolute_error(y_true, y_pred)
scoring_dict = {'RMSE' : metrics.make_scorer(RMSE),
                'MAE' : 'neg_mean_absolute_error',
                'R2': 'r2'}

#scoring = 'neg_mean_squared_error' # MSE ** 2
#scoring = 'r2' # R ** 2
#scoring ='neg_mean_absolute_error' # MAE
#"gamma": np.linspace(.001,0.1, 20)})
# https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search

#kr_opti = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.5), cv=5,
                       #param_grid={"alpha": [np.linspace(1, 5, 10)]}, 
                       #scoring = scoring,
                       #scoring = scoring_dict,
                       #refit = 'MSE',
                       #verbose = True,
                       #return_train_score=True)
                                
#kr_opti.fit(X_hold_out_set,Y_hold_out_set)
#print("score KRR OTIMISEE %.3f" % kr_opti.score(X_hold_out_set,Y_hold_out_set) )
#results = kr_opti.cv_results_
#print("results", results)


RMSE_SCORE = []
MAE_SCORE = []
R2_SCORE = []

min_RMSE = 1e6 
alpha_0 = 3e-2
gamma_0 = 3e-2
alpha = alpha_0
gamma = gamma_0
h = 1e1 # pas de descente du gradient
kr = KernelRidge(kernel='rbf', gamma = gamma, alpha = alpha)
kr.fit(X_training_set,Y_training_set)         
Y_kr_pred = kr.predict(X_hold_out_set)
RMSE_SCORE.append(RMSE(Y_hold_out_set,Y_kr_pred))

DKR_SCORE_alpha = 1e6
DKR_SCORE_gamma = 1e6
while (DKR_SCORE_gamma**2 + DKR_SCORE_alpha**2) > 1e-6:
    
    print((alpha,gamma),'score RMSE =',RMSE(Y_hold_out_set,Y_kr_pred))
    
    KR_SCORE = RMSE_SCORE[-1] 
    
    kr_gamma_h = KernelRidge(kernel='rbf', gamma = gamma + h, alpha = alpha)
    kr_gamma_h.fit(X_training_set,Y_training_set)         
    Y_kr_pred_gamma_h = kr_gamma_h.predict(X_hold_out_set)
    KR_SCORE_gamma = RMSE(Y_hold_out_set,Y_kr_pred_gamma_h)    
    

    kr_alpha_h = KernelRidge(kernel='rbf', gamma = gamma, alpha = alpha +h)
    kr_alpha_h.fit(X_training_set,Y_training_set)         
    Y_kr_pred_alpha_h = kr_alpha_h.predict(X_hold_out_set)
    KR_SCORE_alpha = RMSE(Y_hold_out_set,Y_kr_pred_alpha_h)
    
    DKR_SCORE_alpha = (KR_SCORE_alpha - KR_SCORE)/h
    DKR_SCORE_gamma = (KR_SCORE_gamma - KR_SCORE)/h
    
    gamma -= h * DKR_SCORE_gamma
    alpha -= h * DKR_SCORE_alpha
    
    kr = KernelRidge(kernel='rbf', gamma = gamma, alpha = alpha)
    kr.fit(X_training_set,Y_training_set)         
    Y_kr_pred = kr.predict(X_hold_out_set)
    RMSE_SCORE.append(RMSE(Y_hold_out_set,Y_kr_pred))    
    
    #MAE_SCORE.append(MAE(Y_hold_out_set,Y_kr_pred))

def plot_alpha():    
    fig, ax = plt.subplots()
    ax.plot(alpha_grid,RMSE_SCORE,label = 'RMSE',color = 'r',marker='o')
    #ax.set_ylim(min(RMSE_SCORE)-200,max(RMSE_SCORE)+200)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('score')
    ax.set_title(r'performance en fonction du parametre $\alpha$ pour $\gamma$ = 1e-2')
    plt.show()
    return None




# remplissage sur la grille a partir des scores calcules

def plot_mesh():
    X_mesh, Y_mesh = np.meshgrid(gamma_grid_log2, alpha_grid_log2)    
    Z_mesh_RMSE = np.zeros(X_mesh.shape)
    k = 0
    for i in range(Z_mesh_RMSE.shape[0]):
        for j in range(Z_mesh_RMSE.shape[1]):
            Z_mesh_RMSE[i][j] = RMSE_SCORE[k]
            k += 1 
    fig, ax = plt.subplots()  
    #ax.set_ylim(log2(min(alpha_grid))-200,log2(max(alpha_grid))+200)
    #ax.set_xlim(log2(min(gamma_grid))-200,log2(max(gamma_grid))+200)
    CS = ax.contour(X_mesh, Y_mesh, Z_mesh_RMSE,1000)
    
                    
    cp = ax.contour(X_mesh, Y_mesh, Z_mesh_RMSE,colors='black', linestyles='dashed')
    ax.clabel(cp, inline=True, fontsize=10)    
    #x.set_title('Simplest default with labels')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\alpha$')
    fig.colorbar(CS)
    plt.show()   


# ######################## performance modele ##################################
def perform():
    kr_final = KernelRidge(kernel='rbf', gamma = gamma_min, alpha = alpha_min)
    kr_final.fit(np.concatenate((X_hold_out_set,X_training_set)),
                 np.concatenate((Y_hold_out_set,Y_training_set)))     
    Y_kr_pred_final = kr_final.predict(X_validation)
    print('score final RMSE =',RMSE(Y_validation,Y_kr_pred_final))
    
    print('temps execution = ',time.time() - start_time)
        
