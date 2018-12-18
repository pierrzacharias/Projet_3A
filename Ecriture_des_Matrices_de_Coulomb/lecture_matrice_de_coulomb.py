#<<<<<<<<<<<<<<<<< Lecture matrices de Coulombs >>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import os as os
import pickle
matrice_coulomb = open('number_of_non_H_atoms.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
energie_atomisation = open('energie_atomisation.txt','rb')
energie_atomisation_depickler = pickle.Unpickler(energie_atomisation)
def lecture():
    #Coulomb_matrix_i = matrice_coulomb_depickler.load()
    return matrice_coulomb_depickler.load()
#matrice_coulomb.close()
#energie_atomisation.close()
