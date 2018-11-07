#<<<<<<<<<<<<<<< scipt pour tester >>>>>>>>>>>>>>>>>>>>

import numpy as np
matrice_coulomb_i = np.zeros((5,5))
k = 1
#numeros = np.array([[1],[2],[3],[4],[5]])
#print(numeros)
# generation d'une matrice symetrique
for i in range(5): 
    for j in range(i):
        if i == j:
            matrice_coulomb_i[i][j] = 0
        matrice_coulomb_i[i][j] = k
        matrice_coulomb_i[j][i] = k
        k += 1
print(matrice_coulomb_i)

# comme matrice_coulomb_i est symetrique on garde que la partie inférieure de la matrice 
# que l'on place dans un vecteur colonne 
#

# on teste le réarangement sur une petite matrice 
Coulomb_matrix_c = matrice_coulomb_i
for k in range(1,5): 
    print('new k = ',k)
    temp = Coulomb_matrix_c[k].copy()
    print('temp = ',temp)
    j = k 
    while j > 0 and np.linalg.norm(temp) > np.linalg.norm(Coulomb_matrix_c[j-1]): 
        
        Coulomb_matrix_c[j] = Coulomb_matrix_c[j-1] 
        print('>>>>>>>>>>>>>>>>>> j =', j)
        print(Coulomb_matrix_c)
        # permutation des colonnes des lignes subsitituées pour garder la propriété de symétrie de la matrice 
        stockage1 = Coulomb_matrix_c[:,j].copy()
        stockage2 = Coulomb_matrix_c[:,j-1].copy()
        
        Coulomb_matrix_c[:,j-1] = stockage1
        Coulomb_matrix_c[:,j] = stockage2
        temp[j-1], temp[j] = temp[j],temp[j-1]
        print('temp = ',temp)
        numeros[j-1], numeros[j] = numeros[j], numeros[j-1]
      
        print('apres switch')
        print(Coulomb_matrix_c)
        #print(numeros)        
        
        j -= 1 
        
        
    Coulomb_matrix_c[j] = temp 
    print(Coulomb_matrix_c)

matrice_coulomb_rearange = np.zeros((20,1))
k = 0
for i in range(5): 
    for j in range(i+1):
        matrice_coulomb_rearange[k] = matrice_coulomb_i[i][j]
        k += 1
print(matrice_coulomb_rearange)