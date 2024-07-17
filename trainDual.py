import torch
from readData import *
import numpy as np
from numpy import linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    data = KolendaImageCaption()
    trData = data["train"]
    xs = trData['x']
    V = len(xs)
    ksas = [{"gamma":0.01} for i in range(0,V)]
    K = [kernelMtxRBF(xs[i],ksas[i],bar=True) for i in range(0,V)]
    Omega = [centreMtx(K[i],bar=True) for i in range(0,V)]
    
def kernelMtxRBF(x,ksas,bar=False): # non-centered
    # ksas: kernel-specific arguments. dict.
    # here: gamma
    gamma = ksas["gamma"]
    size = len(x)
    K = np.zeros((size,size))
    topIter = range(0,size)
    if bar:
        topIter = tqdm(topIter)
        topIter.set_description("kernelMtx")
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
