################################################################################
# <<<<<<<<<<<<<<<<< use KernelRIdge methods >>>>>>>>>>>>>>>>>>>>>>>>>>>
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


# ######### lecture des données ############################
matrice_coulomb = open('matrice_coulomb.txt', 'rb')
matrice_coulomb_depickler = pickle.Unpickler(matrice_coulomb)
def lecture():
    Coulomb_matrix_i = matrice_coulomb_depickler.load()
    return Coulomb_matrix_i







