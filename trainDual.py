import torch
from readData import *
import numpy as np
from numpy import linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
import random

def main():

    # hyper:
    gam = 0.1
    rho = 0.3
    eta = 1.0
    hyper = [gam, rho, eta]

    data = kolendaImageCaption()
    
    trData = data["train"]
    xData = trData['x']
    V = len(xData)
    
    fold = 5
    h = multiViewCross(trData,fold,bar=True)
    
    # for i in range(fold):
        # f = h[i]
        # for v in range(V):
            # print( len( f["val"]['x'][v] ), end='\t')
        # print()
    
    # uit = trainDualTensorRBF(xData,hyper,bar=True)
    
    # sp.io.savemat("dSprites.mat",{"OmegaAdd":OmegaAdd,"OmegaMul":OmegaMul,"gamma":gam})

def multiViewCross(data,fold,bar=False):
    """
    []          - folds
    {}          - train/val
    {}          - x/y
    []          - views
    np.array    - data
    """
    xData = data['x']
    yData = data['y']
    V = len(xData)
    n = len(xData[0])
    gts = list(range(fold))
    idxs = []
    while len(idxs) < n:
        idxs = idxs + gts
    hindCut = len(idxs) - n
    idxs = idxs[0:(len(idxs)-hindCut)]
    random.shuffle(idxs)
    flds = []
    top = range(fold)
    if bar:
        top = tqdm(top)
        top.set_description("Cross val.")
    for w in top: # folds
        # data met index w gaat naar validatie
        xtr = []
        xva = []
        ytr = []
        yva = []
        for v in range(V): # views
            axv = xData[v] # all x-data in current view
            ayv = yData # all y-data (indep. of view!)
            xtrv = [] # training x-data in current view
            ytrv = [] # training y-data in current view
            xvav = [] # validation x-data in current view
            yvav = [] # validation y-data in current view
            for i in range(n): 
                if (idxs[i] != w):
                    xtrv.append(axv[i]) # add x training
                    ytrv.append(ayv[i]) # add y training
                else:
                    xvav.append(axv[i]) # add x validation
                    yvav.append(ayv[i]) # add y validation
            # make np arrays
            xtrva = np.array(xtrv)
            xvava = np.array(xvav)
            ytrva = np.array(ytrv)
            yvava = np.array(yvav)
            # add view into data
            xtr.append(xtrva)
            xva.append(xvava)
            ytr.append(ytrva)
            yva.append(yvava)
        # pack-up
        val   = {'x':yva,'y':yva}
        train = {'x':ytr,'y':ytr}
        fl = {"train":train,"val":val}
        flds.append(fl) # add to folds list
    return flds
    
def trainDualTensorRBF(xData,hyper,bar=False):
    gam = hyper[0]
    rho = hyper[1]
    eta = hyper[2]
    xs  = xData
    V = len(xs)
    ksas = [{"gamma":gam} for i in range(0,V)]
    top = range(0,V)
    if bar:
        top = tqdm(top)
        top.set_description("Kernelmtx")
    K = [kernelMtxRBF(xs[i],ksas[i]) for i in top]
    top = range(0,V)
    if bar:
        top = tqdm(top)
        top.set_description("Centre K")
    Omega = [centreMtx(K[i]) for i in top]
    N = K[0].shape[0]
    OmegaAdd = np.zeros((N,N))
    for i in Omega:
        OmegaAdd += i
    OmegaMul = np.ones((N,N))
    for i in Omega:
        OmegaMul = OmegaMul * i
    OmegaTot = (1 - rho)*OmegaAdd + rho*OmegaMul
    L,S = torch.linalg.eigh(torch.from_numpy(OmegaTot/eta))
    uit = { "eigenvalues":L,
            "eigenvectos":S,
            "K":K, "Omega":Omega,
            "OmegaAdd":OmegaAdd,
            "OmegaMul":OmegaMul,
            "OmegaTot":OmegaTot}
    return uit
    
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
