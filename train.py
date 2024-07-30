from scipy.io import loadmat
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
import numpy as np
from scipy.linalg import eigh

NJOBS = None

def main():
    # for tests

    # Kolenda dataset
    path = Path(".")
    path = path / "data" / "KolendaImageCaption" / "Kolenda_withNoise.mat"
    mat = loadmat(path)
    xs = mat["X"][0]
    trainDual(xs,"rbf",0.6,1,gamma=1)

def trainDual(xs,kernel,rho,eta,**kwargs):
    """
    INPUT:
    xs:     list of views
    kernel: kernel as in scikit-learn; [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
    rho:    weighing of KPCA-ADD and KPCA-PROD in KPCA-ADDPROD
    eta:    scale variable in the eigenvalue problem
    kwargs are passed straight to the kernel function
    OUTPUT:
    lambdas: eigenvalues of KPCA-model, increasing order
    hs:      eigenvectors of KPCA-model
    """
    V = len(xs) # number of views
    # setup matrices
    Ks = [] # list of kernels per view (unneeded)
    Omegas = [] # centered kernels
    for v in range(V): # find kernels
        K = pairwise_kernels(xs[v],xs[v],kernel,n_jobs=NJOBS,**kwargs)
        Ks.append(K) # (unneeded)
        trsfr = KernelCenterer().fit(K)
        Omegas.append( trsfr.transform(K))
    # find procuct and -sum matrices
    OmegaAdd  = sum(Omegas)
    OmegaProd = np.prod( np.array(Omegas), axis=0 )
    # find end matrix
    OmegaTot = ((1-rho) * OmegaAdd) + (rho * OmegaProd)
    # eigenvalue problem: KPCA
    lambdas,hs = eigh((1/eta) * OmegaTot)
    return lambdas, hs

main()
