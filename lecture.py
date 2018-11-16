#<<<<<<<<<<<<<<<<< Lecture matrices de Coulombs >>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import os as os
import pickle
matrice_coulomb = open('matrice_coulomb.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
def lecture():
    Coulomb_matrix_i = matrice_coulomb_depickler.load()
    return Coulomb_matrix_i
