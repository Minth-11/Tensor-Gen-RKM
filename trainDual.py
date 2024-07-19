import torch
from readData import *
import numpy as np
from numpy import linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp

def main():

    # hyper:
    rho = 0.3
    eta = 1.0
    gam = 0.1

    # data = kolendaImageCaption()
    data = dSprites(bar=True)
    trData = data["train"]
    xs = trData['x']
    # print(xs[0][0])
    # print(torch.is_sparse(xs[0][0]))
    V = len(xs)
    ksas = [{"gamma":gam} for i in range(0,V)]
    K = [kernelMtxRBF(xs[i],ksas[i]) for i in tqdm(range(0,V))]
    Omega = [centreMtx(K[i]) for i in tqdm(range(0,V))]
    N = K[0].shape[0]
    OmegaAdd = np.zeros(N,N)
    for i in Omega:
        OmegaAdd += i
    OmegaMul = np.ones((N,N))
    for i in Omega:
        OmegaMul = OmegaMul * i
    OmegaTot = (1 - rho)*OmegaAdd + rho*OmegaMul
    L,S = torch.linalg.eigh(torch.from_numpy(OmegaTot/eta))
    # plt.plot([np.log(i) for i in list(L)])
    # plt.show()
    sp.io.savemat("dSprites.mat",{"OmegaAdd":OmegaAdd,"OmegaMul":OmegaMul,"gamma":gam})
   
def kernelMtxRBF(x,ksas,bar=False,prt=''): # non-centered
    # ksas: kernel-specific arguments. dict.
    # here: gamma
    gamma = ksas["gamma"]
    size = len(x)
    K = np.zeros((size,size))
    topIter = range(0,size)
    if bar:
        topIter = tqdm(topIter)
        topIter.set_description("kernelMtx")
    else:
        for i in topIter:
            for j in range(0,size):
                K[i,j] = np.exp( -1*gamma*la.norm(x[i] - x[j]) )
    return K
    
def centreMtx(K,bar=False):
    N = K.shape[0]
    horSums = K.sum(axis=0)
    verSums = K.sum(axis=0)
    totSum  = K.sum()
    tss = totSum / (N^2)
    Omega = np.zeros((N,N))
    topIter = range(0,N)
    if bar:
        topIter = tqdm(topIter)
        topIter.set_description("centring")
    for i in topIter:
        for j in range(0,N):
            Omega[i][j] = K[i][j] - ((horSums[i]+verSums[j])/N) + tss
    return Omega

main()
