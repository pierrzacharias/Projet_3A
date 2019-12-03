################################################################################
#   lecrture des donnees des simulation enrengistre dans des fichiers textes
################################################################################
import os
#os.chdir("C:/Users/pierre gauthier/Documents/3A/Projet_3A/Methode_Kernel_Ridge_Regression")
os.chdir("C:/Users/pierr/Documents/3A/projet/Projet_3A/Methode_Kernel_Ridge_Regression")

import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import log2
matrice_RMSE = open('Resultat_RMSE_RBF_2_30_10.txt', 'rb')
matrice_RMSE_depickler = pickle.Unpickler(matrice_RMSE)
matrice_R2 = open('Resultat_R2_RBF_2_30_10.txt', 'rb')
matrice_R2_depickler = pickle.Unpickler(matrice_R2)
matrice_MAE = open('Resultat_MAE_RBF_2_30_10.txt', 'rb')
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
alpha_grid_log2 = np.arange(-30,-10,0.5)
alpha_grid = [2.**alpha_grid_log2[i] for i in range(len(alpha_grid_log2))]
gamma_grid_log2 = np.arange(-30,-10,0.5)
#gamma_grid_log2 = [-15]
gamma_grid = [2.**gamma_grid_log2[i] for i in range(len(gamma_grid_log2))]


######################### pour RMSE ############################################
X_mesh, Y_mesh = np.meshgrid(gamma_grid_log2, alpha_grid_log2)    
Z_mesh_RMSE = np.zeros(X_mesh.shape)
k = 0
for i in range(Z_mesh_RMSE.shape[0]):
    for j in range(Z_mesh_RMSE.shape[1]):
        Z_mesh_RMSE[i][j] = RMSE_SCORE[k]
        k += 1 
#print('temps remplissage grille = ',time.time() - start_time)

import matplotlib.colors as mc
def addNorm(cmapData):
    cmapData['norm'] = mc.BoundaryNorm(cmapData['bounds'], cmapData['cmap'].N)
    return True
def discretize(cmap, bounds):
    resCmap = {}
    resCmap['cmap'] = mc.ListedColormap( \
        [cmap(i/len(bounds[1:])) for i in range(len(bounds[1:]))]
    )
    resCmap['bounds'] = bounds
    addNorm(resCmap)
    return resCmap

levels = sorted(set(RMSE_SCORE))
levels = [levels[i] for i in range(0,len(levels),60) if levels[i] <1]
#levels = sorted(set(levels))
levels = levels + [RMSE_SCORE[i] for i in range(0,len(RMSE_SCORE),100)]
levels = sorted(set(levels))
#levels = [i * 1e6 for i in levels]
fig, ax = plt.subplots()  
cmapData = discretize(plt.cm.jet, bounds=levels)
cp = ax.contourf(X_mesh, Y_mesh, Z_mesh_RMSE,levels=levels,linestyles = 'dashed',cmap=cmapData['cmap'], norm=cmapData['norm'],linewidths=1)
ax.contour(cp)
#ax.clabel(cp, inline=True,levels=levels, fontsize=10,color='b')    
ax.grid(c='k', ls='-', alpha=0.7)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
#thicks = [i  for i in levels]
cbar = fig.colorbar(cp, orientation='horizontal')
cbar.ax.set_xticklabels([f'{i * 1e6:.0f}' for i in levels])
cbar.ax.set_xlabel('$10^{-6}$ RMSE')
#cbar_ticks = levels
#cbar.set_ticks(cbar_ticks)
ax.plot([-13], [-29], 'k.', markersize=25.0)
plt.show()   

# affichage p color

#fig1, ax1 = plt.subplots()  
#cs = ax1.pcolor(X_mesh, Y_mesh, Z_mesh_RMSE, edgecolors='k', linewidths=1)
#fig.colorbar(cs)
#cbar.ax.set_ylabel('RMSE')
#plt.show()

######################## pour MAE ##############################################
 
Z_mesh_MAE= np.zeros(X_mesh.shape)
k = 0
for i in range(Z_mesh_MAE.shape[0]):
    for j in range(Z_mesh_MAE.shape[1]):
        Z_mesh_MAE[i][j] = MAE_SCORE[k]
        k += 1 
#print('temps remplissage grille = ',time.time() - start_time)

import matplotlib.colors as mc
def addNorm(cmapData):
    cmapData['norm'] = mc.BoundaryNorm(cmapData['bounds'], cmapData['cmap'].N)
    return True
def discretize(cmap, bounds):
    resCmap = {}
    resCmap['cmap'] = mc.ListedColormap( \
        [cmap(i/len(bounds[1:])) for i in range(len(bounds[1:]))]
    )
    resCmap['bounds'] = bounds
    addNorm(resCmap)
    return resCmap

levels = sorted(set(MAE_SCORE))
levels = [levels[i] for i in range(0,len(levels),60) if levels[i] <1]
#levels = sorted(set(levels))
levels = levels + [MAE_SCORE[i] for i in range(0,len(MAE_SCORE),100)]
levels = sorted(set(levels))
#levels = [i * 1e6 for i in levels]
fig, ax = plt.subplots()  
cmapData = discretize(plt.cm.jet, bounds=levels)
cp = ax.contourf(X_mesh, Y_mesh, Z_mesh_MAE,levels=levels,linestyles = 'dashed',cmap=cmapData['cmap'], norm=cmapData['norm'],linewidths=1)
ax.contour(cp)
#ax.clabel(cp, inline=True,levels=levels, fontsize=10,color='b')    
ax.grid(c='k', ls='-', alpha=0.7)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
#thicks = [i  for i in levels]
cbar = fig.colorbar(cp, orientation='horizontal')
cbar.ax.set_xticklabels([f'{i * 1e6:.0f}' for i in levels])
cbar.ax.set_xlabel('$10^{-6}$ MAE')
#cbar_ticks = levels
#cbar.set_ticks(cbar_ticks)
ax.plot([-11.5], [-29], 'k.', markersize=25.0)
plt.show()   

########################### pour R2 ############################################
R2_SCORE = [ 1 - i for i in R2_SCORE]

Z_mesh_R2= np.zeros(X_mesh.shape)
k = 0
for i in range(Z_mesh_R2.shape[0]):
    for j in range(Z_mesh_R2.shape[1]):
        Z_mesh_R2[i][j] = R2_SCORE[k]
        k += 1 
#print('temps remplissage grille = ',time.time() - start_time)

import matplotlib.colors as mc
def addNorm(cmapData):
    cmapData['norm'] = mc.BoundaryNorm(cmapData['bounds'], cmapData['cmap'].N)
    return True
def discretize(cmap, bounds):
    resCmap = {}
    resCmap['cmap'] = mc.ListedColormap( \
        [cmap(i/len(bounds[1:])) for i in range(len(bounds[1:]))]
    )
    resCmap['bounds'] = bounds
    addNorm(resCmap)
    return resCmap

levels = sorted(set(R2_SCORE))
levels = [levels[i] for i in range(0,len(levels),50) if levels[i] > 0.99999999]
#levels = sorted(set(levels))
levels = levels + [R2_SCORE[i] for i in range(0,len(R2_SCORE),100)]
levels = [max(levels)]+ levels
levels = levels + list(np.linspace(0.99,max(R2_SCORE),5))
levels = sorted(set(levels))

#levels = [i * 1e6 for i in levels]
fig, ax = plt.subplots()  
cmapData = discretize(plt.cm.jet, bounds=levels)
cp = ax.contourf(X_mesh, Y_mesh, Z_mesh_R2,levels=levels,linestyles = 'dashed',cmap=cmapData['cmap'], norm=cmapData['norm'],linewidths=1)
ax.contour(cp)
#ax.clabel(cp, inline=True,levels=levels, fontsize=10,color='b')    
ax.grid(c='k', ls='-', alpha=0.7)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
#thicks = [i  for i in levels]
cbar = fig.colorbar(cp, orientation='horizontal')
cbar.ax.set_xticklabels([f'{i :.5f}' for i in levels])
cbar.ax.set_xlabel(' R2')
#cbar_ticks = levels
#cbar.set_ticks(cbar_ticks)
ax.plot([-13], [-29], 'k.', markersize=25.0)
plt.show()  