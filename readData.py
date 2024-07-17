import torch
from torchvision import datasets, transforms
from scipy.io import loadmat
from pathlib import Path

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

def KolendaImageCaption():
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
