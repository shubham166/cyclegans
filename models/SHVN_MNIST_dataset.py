import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm
import torchvision
from pytorch_lightning import LightningDataModule

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
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'test': transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
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
        # return 100

    def __getitem__(self, index):
        return {
            "A": self.dataA[index],
            "B": self.dataB[index]
        }


class ShvnMnistDataModule(LightningDataModule):
    def __init__(self,
        data_path: str = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    
    def setup(self, stage=None) -> None:
        self.train_dataset = ShvnMnistDataset("./Data", "train")
        self.val_dataset = ShvnMnistDataset("./Data", "test")
        self.test_dataset = ShvnMnistDataset("./Data", "test")

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )






