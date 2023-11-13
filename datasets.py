from torchvision.datasets import CIFAR10
import torchvision


class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, download=download)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

class CustomGTSRB(torchvision.datasets.ImageFolder): 

    def __init__(self, root,transform=None, target_transform=None):
        super().__init__(root,transform)
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

