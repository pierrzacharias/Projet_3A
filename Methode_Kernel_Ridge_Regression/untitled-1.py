import pickle
matrice_RMSE= open("Resultat_RMSE_RBF.txt","wb")
matrice_RMSE_pickler = pickle.Pickler(matrice_RMSE)
for i in range(len(RMSE_SCORE)):
    matrice_RMSE_pickler.dump(RMSE_SCORE[i])
matrice_RMSE.close()
matrice_R2 = open("Resultat_R2_RBF.txt","wb")
matrice_R2_pickler = pickle.Pickler(matrice_R2)
for i in range(len(R2_SCORE)):
    matrice_R2_pickler.dump(R2_SCORE[i])
matrice_R2.close()
matrice_MAE= open("Resultat_MAE_RBF.txt","wb")
matrice_MAE_pickler = pickle.Pickler(matrice_MAE)
for i in range(len(MAE_SCORE)):
    matrice_MAE_pickler.dump(MAE_SCORE[i])
matrice_MAE.close()
