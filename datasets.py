import pickle
import os
import numpy as np
import torch
import platform
from torch.utils.data import Dataset

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))
def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    del X, Y
    X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y
    
class CIFAR10_dataset(Dataset):
    """CIFAR10 dataset."""

    def __init__(self,root_dir, transform=None, phase = ""):
        """
        Arguments:
            root_dir (string): Directory with all the downloaded dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        self.X = {}
        self.Y = {}
        
        X_train, Y_train, X_test, Y_test = load_CIFAR10(root_dir)
        
        self.X["train"] = X_train
        self.X["test"] = X_test
        
        self.Y["train"] = Y_train
        self.Y["test"] = Y_test
        
    def __len__(self):
            return len(self.X[self.phase])
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.X[self.phase][idx]
        image = image.reshape(3,32,32)
        image = image.transpose(1,2,0)

        y = self.Y[self.phase][idx]
        y = np.array(y)


        if self.transform:
            image = self.transform(image)

        return image,y,idx
