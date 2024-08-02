import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from train import trainDual
import torch

# NCLASSES = 125
NCLASSES = 15

def main():
    skr = "/volume1/scratch/zopdebee/databases/Sketchy/rendered_256x256/256x256/sketch/tx_000000000000"
    phr = "/volume1/scratch/zopdebee/databases/Sketchy/rendered_256x256/256x256/photo/tx_000000000000"
    data = TripleDataset(phr,skr)
    # print(data.__getitem__(1111))
    # mn = 3000
    # for i in tqdm(range(data.len)):
    #     h = data.__getitem__(i)
    #     if h['L'] < mn:
    #         mn = h['L']
    #         print(mn) # -> 125 klassen, incl. 0
    xsv = [[],[],[]]
    # 0: photo
    # 1: sketch
    # 2: labal (one-hot)
    for i in tqdm(range(data.len)):
        item = data.__getitem__(i)
        if (item['L'] < NCLASSES): # filter out other classes
            # photo
            h = torch.reshape(item['P'],(-1,))
            xsv[0].append(h.numpy())
            # sketch
            h = torch.reshape(item['S'],(-1,))
            xsv[1].append(h.numpy())
            # label
            h = np.zeros(NCLASSES)
            h[item['L']] = 1.0
            xsv[2].append(h)
    print("Data size:" + str(len(xsv[0])))
    xsv = [np.array(i) for i in xsv]
    lambdas, hs = trainDual(xsv,"rbf",0.6,1,gamma=1)
    print("KLAAR")
    input()


def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idex

def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images


class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root):
        super(TripleDataset, self).__init__()
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        classes, class_to_idx = find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root

        self.photo_paths = sorted(make_dataset(self.photo_root))
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, label = self._getrelate_sketch(photo_path)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        P = self.tranform(photo)
        S = self.tranform(sketch)
        L = label
        return {'P': P, 'S': S, 'L': L}

    def __len__(self):
        return self.len

    def _getrelate_sketch(self, photo_path):

        paths = photo_path.split('/')
        fname = paths[-1].split('.')[0]
        cname = paths[-2]

        label = self.class_to_idx[cname]

        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))

        sketch_rel = []
        for sketch_name in sketchs:
            if sketch_name.split('-')[0] == fname:
                sketch_rel.append(sketch_name)

        rnd = np.random.randint(0, len(sketch_rel))

        sketch = sketch_rel[rnd]

        return os.path.join(self.sketch_root, cname, sketch), label

main()
