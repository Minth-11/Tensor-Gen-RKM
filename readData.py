import torch
from torchvision import datasets, transforms
from scipy.io import loadmat
from pathlib import Path
from disentanglement_datasets import DSprites

"""
Algemeen formaat, van buiten naar binnen:
dict: train/test
dict: X/Y
lijst: views
np-array: data
"""

# def mnistDataLoader():
    # train_loader = torch.utils.data.DataLoader(
                        # datasets.MNIST('./data', 
                                        # train=True,
                                        # transform=transforms.ToTensor(),
                                        # )   )
                                        
    # test_loader = torch.utils.data.DataLoader(
                        # datasets.MNIST('./data', 
                                        # download=True, 
                                        # train=False,
                                        # transform=transforms.ToTensor(),
                                        # )   )
    
    # return {"train":train_loader,"test":test_loader}

def kolendaImageCaption():
    path = Path("data\KolendaImageCaption\Kolenda_withNoise.mat")
    mat = loadmat(path)
    """
    __header__
    __version__
    __globals__
    X
    Y
    """
    # voorlopig geen test TODO
    # print(mat["X"])
    trys = mat["Y"]
    trxs = [i for i in mat["X"][0]]
    train = {'x':trxs,'y':trys}
    test = {}
    uit = {"train":train,"test":test}
    return uit

def dSprites(bar=False):
    data = DSprites(root="./data",download=True)
    lCOLOUR = 0 # irrelevant
    lSHAPE = 1
    lSCALE = 2
    lORIENT = 3
    lPOSX = 4
    lPOSY = 5
    xDict = {}
    yDict = {}
    from tqdm import tqdm
    tst = lSCALE
    top = data
    if bar:
        top = tqdm(top)
        top.set_description("dSprites")
    for i in top:
        a = i["latent"][lPOSX].item()
        b = i["latent"][lPOSY].item()
        if (a,b) in xDict:
            xDict[(a,b)].append(torch.reshape(i['input'],(-1,)))
            yDict[(a,b)].append(i['latent'])
        else:
            xDict[(a,b)] = [torch.reshape(i['input'],(-1,))]
            yDict[(a,b)] = [i['latent']]
    xs = list(map(lambda x: x[1],xDict.items()))
    ys = list(map(lambda x: x[1],yDict.items()))
    train = {'x':xs,'y':ys}
    test = {}
    uit = {"train":train,"test":test}
    return uit
    