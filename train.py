from scipy.io import loadmat
from pathlib import Path

def main():
    # for tests

    # Kolenda dataset
    path = Path("data\KolendaImageCaption\Kolenda_withNoise.mat")
    mat = loadmat(path)
    print(mat)

def trainDual(xs,kernel):
    """
    xs: list of views
    kernel: kernel
    rho: ratio
    """
    pass

main()
