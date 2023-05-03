import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm
import torchvision


data_transforms = {
    "svhn": {
        'train': transforms.Compose([
            transforms.Resize(286, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    },
    "mnist": {
       'train': transforms.Compose([
            transforms.Resize(286, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([.5], [.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5], [.5])
        ]) 
    }
}



class ShvnMnistDataset(Dataset):
    def __init__(self, root, mode):
        assert mode in 'train test'.split(), 'mode should be either train or test'
        
        super().__init__()
        self.transformA = data_transforms["svhn"][mode]
        self.transformB = data_transforms["mnist"][mode]
        self.dataA = torchvision.datasets.SVHN(os.path.join(root, "svhn"), split=mode, download=True, transform=self.transformA)

        self.dataB = torchvision.datasets.MNIST(os.path.join(root, "mnist"), train=(mode=="train"), download=True, transform=self.transformB)
    
    def __len__(self):
        return min(self.dataA.__len__(), self.dataB.__len__())
    
    def __getitem__(self, index):
        return {
            "A": self.dataA[index],
            "B": self.dataB[index]
        }



