#<<<<<<<<<<<<<<<<< ecriture des matrices de Coulombs >>>>>>>>>>>>>>>>>>>>>>>>>>>

################################################################################
import numpy as np
import os as os
import pickle
#os.chdir("C:/Users/pierr/Documents/3A/projet/Projet_3A/Ecriture_des_Matrices_de_Coulomb")
################################################################################


# pour chaque molecule :
# ligne 1 : energie d'atomisation
# autre ligne coordonnees des atomes (pourquoi 2 sets de coordonnée ? pour quoi ne commence pas a zero ?)


# tableau periodique
AbbrIndex = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Uut": 113, "Fl": 114, "Uup": 115, "Lv": 116, "Uus": 117, "Uuo": 118}

# ouverture fichier xyz contenant les données
fichier_xyz = open("dsgdb7ae2.xyz","r")
contenu_fichier_xyz = fichier_xyz.read()    
contenu_fichier_xyz = contenu_fichier_xyz.splitlines()
 
# creation fichier matrice coulomb dans lequel on ecrit les matrices 
matrice_coulomb = open("matrice_coulomb.txt","wb")
matrice_coulomb_pickler = pickle.Pickler(matrice_coulomb)

# creation fichier energie_atomisation dans lequel on stocke les énergie d'atomisation dans le meme ordre
energie_atomisation = open("energie_atomisation.txt","wb")
energie_atomisation_pickler = pickle.Pickler(energie_atomisation)


# creation fichier number_of_non_H_atoms dans lequel on stocke le nombre d'atomes non-H
number_of_non_H = open("number_of_non_H_atoms.txt","wb")
number_of_non_H_pickler = pickle.Pickler(number_of_non_H)

###############################################################
# nous parcourons le fichier molécule par molécule et on calcule la matrice de Coulomb associé,
# que l'on réarange par norme descendante et dont on ne garde que la partie inférieure car symétrique
###############################################################
roam = 0
while roam < len(contenu_fichier_xyz):
    
    # on utilise le nombre d'atome de chaque molécule pour parcourir le fichier en sautant l'energie d'atomisation 
    number_of_atom = int(contenu_fichier_xyz[roam])
    number_of_non_H_atoms = 0 # on compte les atomes non H
    energie_atomisation_c = float(contenu_fichier_xyz[ roam + 1 ].split()[1])
    ###########################################################
    # nous parcourons les atomes pour une molécule 
    ###########################################################
    
    list_atomic_number = [] # liste des numéros atomiques
    list_positions = [] # liste des positions des atomes
    
    for atom in range(number_of_atom): # on saute l'energie d'atomisation 
        
        current_atom_xyz = contenu_fichier_xyz[atom + roam + 2]
        current_atom_xyz =  current_atom_xyz.split()
        list_atomic_number.append( AbbrIndex[ current_atom_xyz[0]] ) # utilisation dictionnaire pour trouver le numero atmique
        
        list_positions.append([float(current_atom_xyz[4]), float(current_atom_xyz[5]), float(current_atom_xyz[6])] )# calcul du R 
        if current_atom_xyz[0] != 'H': number_of_non_H_atoms += 1
    #########################################################
    # calcul Coulomb Matrix 
    #########################################################
    
    Coulomb_matrix_c = np.zeros((23,23))
    for i in range(number_of_atom):
        for j in range(number_of_atom):
                if i == j:
                    Coulomb_matrix_c[i][j] = 0.5 * list_atomic_number[i] ** 2.4
                else :
                    Coulomb_matrix_c[i][j] = float(list_atomic_number[i]) * float(list_atomic_number[j]) / ( ((list_positions[i][0] - list_positions[j][0])**2 + (list_positions[i][1] - list_positions[j][1])**2 + (list_positions[i][2] - list_positions[j][2])**2)  ** (0.5) )      
    # print(Coulomb_matrix_c[0:number_of_atom][0:number_of_atom])
    
    #########################################################
    # reogarnisation de la matrice par norme descendante des lignes pour que la matrice ne dépende pas de l'atome de départ considéré
    # on utilise l'algorithme de tri naïf classique
    #########################################################
    for k in range(1,number_of_atom): 
        
        temp = Coulomb_matrix_c[k].copy()
        j = k 
        # on test si la norme au dessus est supérieure ou inférieure
        while j > 0 and np.linalg.norm(temp) > np.linalg.norm(Coulomb_matrix_c[j-1]): 
            
            Coulomb_matrix_c[j] = Coulomb_matrix_c[j-1] 
            
            # permutation des colonnes des lignes subsitituées pour garder la propriété de symétrie de la matrice et ainsi ne pas perdre d'information
            # j'utilise des variable de stockage car a,b=b,a ne marche ici je ne sais pas pourquoi
            stockage1 = Coulomb_matrix_c[:,j].copy()
            stockage2 = Coulomb_matrix_c[:,j-1].copy()
            Coulomb_matrix_c[:,j-1] = stockage1
            Coulomb_matrix_c[:,j] = stockage2
            
            temp[j-1], temp[j] = temp[j],temp[j-1]
            j -= 1 
            
        Coulomb_matrix_c[j] = temp     
        
        
        norm_i = np.linalg.norm(Coulomb_matrix_c[i])
    # print(Coulomb_matrix_c[0:number_of_atom][0:number_of_atom])
    
    ###################################################################
    # on ne garde que la partie inférieure de la matrice car symetrique
    ###################################################################
    Coulomb_matrix_vector = [0] * 276
    k = 0
    # on remplit la matruce de Coulomb de taille maximale et on garde les zeros dans le vecteurs prenant la partie inférieure de la matrice 
    # pour conserver les mêmes informations sur les atomes qui ont la mêm configuration entre les molécules 
    for i in range(23): 
        for j in range(i,23):   
            Coulomb_matrix_vector[k] = Coulomb_matrix_c[i][j]
            k += 1   
    ###################################################################
    # ecriture de la matrice de coulomb pour la molécule dans le fichier matrice_coulomb 
    ###################################################################
    matrice_coulomb_pickler.dump(Coulomb_matrix_vector)
    energie_atomisation_pickler.dump(energie_atomisation_c)
    number_of_non_H_pickler.dump(number_of_non_H_atoms)
    ###################################################################
    roam += number_of_atom + 2 # passage à la prochaine molécule
    ####################################################################


# fermeture des fichiers
fichier_xyz.close()
matrice_coulomb.close()
energie_atomisation.close()

    