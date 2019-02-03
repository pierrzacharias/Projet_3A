################################################################################
#   lecrture des donnees des simulation enrengistre dans fichier texte
################################################################################

import pickle
import matplotlib.pyplot as plt

matrice_RMSE = open('Resultat_RMSE.txt', 'rb')
matrice_RMSE_depickler = pickle.Unpickler(matrice_RMSE)
matrice_R2 = open('Resultat_R2.txt', 'rb')
matrice_R2_depickler = pickle.Unpickler(matrice_R2)
matrice_MAE = open('Resultat_MAE.txt', 'rb')
matrice_MAE_depickler = pickle.Unpickler(matrice_MAE)
RMSE_SCORE,MAE_SCORE,R2_SCORE = [matrice_RMSE_depickler.load()], [matrice_MAE_depickler.load()],[matrice_R2_depickler.load()]

continuer = True
while continuer == True:
    try:
        RMSE_SCORE.append(matrice_RMSE_depickler.load())
        MAE_SCORE.append(matrice_MAE_depickler.load())
        R2_SCORE.append(matrice_R2_depickler.load())
    except:
        continuer = False    

matrice_RMSE.close()
matrice_R2.close()
matrice_MAE.close()

################################################################################
#                               affichage
################################################################################
alpha_grid_log2 = np.arange(-30,0,0.5)
alpha_grid = [2**alpha_grid_log2[i] for i in range(len(alpha_grid_log2))]
gamma_grid_log2 = np.arange(-30,-10,0.5)
#gamma_grid_log2 = [-15]
gamma_grid = [2**gamma_grid_log2[i] for i in range(len(gamma_grid_log2))]

# remplissage sur la grille a partir des scores calcules


X_mesh, Y_mesh = np.meshgrid(gamma_grid_log2, alpha_grid_log2)    
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