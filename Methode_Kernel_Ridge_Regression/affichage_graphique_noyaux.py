################################################################################
# Ce script propose un affichage des noyaux utilisés pour la regression KKR en1D
################################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
################################################################################
import numpy as np
from sklearn import datasets,model_selection
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


# plt.plot(X_plot, y_kr,color='green', marker='o', linestyle='-',label='KKR regression')



def plot_gamma_gaussian_kernel():
    
    np.random.seed(1)
    N = 20
    X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                        np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]


    X_plot = np.linspace(-6, 6, 1000)[:, None]
    X_src = np.zeros((1, 1))
    plt.figure()
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    fig, ax = plt.subplots()
    for i, gammai in enumerate(np.linspace(0.5,4,10)):
        
        log_dens =  KernelDensity(kernel = 'gaussian',bandwidth = gammai).fit(X_src).score_samples(X_plot)
        a,b = colors.popitem()
        ax.fill(X_plot[:, 0], np.exp(log_dens), a, lw=2)
        
       #ax.legend(('Gamma = %.2f'%gammai,))#,loc = j)
        #axi.xaxis.set_major_locator(plt.MultipleLocator(1))
        #axi.yaxis.set_major_locator(plt.NullLocator())
    ax.legend(['Gamma = %.2f'%gamma for gamma in np.linspace(0.5,4,10)])
    plt.ylim(0.05,0.8)
    plt.xlim(-5,5)
    plt.title('densité du noyau gaussien en fonction de gamma')
    plt.show()
    

    return None    

def plot_kernel():
    
    np.random.seed(1)
    N = 20
    X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                        np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
    
    
    X_plot = np.linspace(-6, 6, 1000)[:, None]
    X_src = np.zeros((1, 1))
    
    fig, ax = plt.subplots(2, 3)#sharex=True, sharey=True)
    #fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)    
    for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                                'exponential', 'linear', 'cosine']):
        axi = ax.ravel()[i]
        log_dens = KernelDensity(kernel = kernel).fit(X_src).score_samples(X_plot)
        axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
        axi.text(-2.6, 0.95, kernel)

        #axi.xaxis.set_major_locator(plt.MultipleLocator(1))
        #axi.yaxis.set_major_locator(plt.NullLocator())
    
        axi.set_ylim(0, 1.05)
        axi.set_xlim(-2.9, 2.9)
    
    ax[0, 1].set_title('Available Kernels')
    plt.show()
    return None
