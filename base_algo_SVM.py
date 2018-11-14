#<<<<<<<<<<<<<<<<<< Ecriture algorithme SVM >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

################################################################################
from sklearn import datasets, model_selection, preprocessing, model_selection, svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
################################################################################

# affiche les resultats graphiquement de l'entrainement du modèle 
def plot_classif_result_SVM(X,y,clf,title):
    # X : parmetres
    # y : observations
    # clf : modele
    # cf: http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    # color maps:
    # couleur du fond
    cmap_light = ListedColormap(['chocolate', 'chartreuse'])
    
    # couleur des points
    cmap_bold = ListedColormap(['lime', 'midnightblue'])    
    
    h = 0.01 # step size in the mesh
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 
    
    # on classifie tout les point du maillage pour l'affichage ensuite
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap_bold, edgecolor = 'k' , s = 20)
    # plot the support vectors:
    plt.scatter(X[clf.support_, 0], X[clf.support_, 1], c = y[clf.support_], cmap = cmap_bold,edgecolor ='k', s=80, marker='*')    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()



# q. 1

# generation dataset
# X_dataset, y_dataset = datasets.make_moons(noise=0.3, n_samples=1000)
# cf : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
X_dataset, y_dataset = datasets.make_blobs( n_features = 2, centers = 2, cluster_std = 1.0, n_samples = 1000)
# décomposition en base de test et base de train cf : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_dataset,y_dataset,test_size=.5)


# affichage dataset train+test
plt.figure()
plt.scatter(X_dataset[:, 0], X_dataset[:, 1], marker='o', c = y_dataset,
            s=20, edgecolor='red')
plt.title('dataset')
plt.show()

# modele SVM linéaire
SVM = svm.SVC( kernel='linear' )
SVM.fit(X_train,y_train)
plot_classif_result_SVM(X_train,y_train,SVM,"SVM")
print("score SVM %.3f" % SVM.score(X_test, y_test) )
# support vectors correspondent bien aux points sur les bords des marges