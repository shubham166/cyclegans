import torch
from models.CycleGan import CycleGan
from models.SHVN_MNIST_dataset import ShvnMnistDataset, ShvnMnistDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

batch_size = 8
accelerator = "cpu"
devices = 1
max_epochs = 2
num_workers = 0
pin_memory = False

root = "./Data"
train = ShvnMnistDataset(root, 'train')
# train = DataLoader(train, batch_size=batch_size, shuffle=True)
# train =  DataLoader(
#             train,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             shuffle=True,
#             pin_memory=pin_memory,
#         )

datamodule = ShvnMnistDataModule("./Data")
model = CycleGan()

trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs, strategy=DDPStrategy(find_unused_parameters=True))

trainer.fit(model, datamodule=datamodule)