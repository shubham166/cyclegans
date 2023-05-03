import torch
from models.CycleGan import CycleGan
from models.SHVN_MNIST_dataset import ShvnMnistDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

batch_size = 8
accelerator = "cpu"
devices = 1
max_epochs = 1
num_workers = 0
pin_memory = False

root = "./Data"
train = ShvnMnistDataset(root, 'train')
# train = DataLoader(train, batch_size=batch_size, shuffle=True)
train =  DataLoader(
            train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
        )
model = CycleGan()

trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs)
trainer.fit(model, train)