import torch
import torchvision
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from torch import nn

device = torch.device('cuda')
print(device)

cifar100_root = '/data/home/yanghao/dataset/cifar/cifar100'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# cifar100_trainset=torchvision.datasets.CIFAR100(cifar100_root,train=True,download=True,transform=transform_train)
cifar100_trainset=torchvision.datasets.CIFAR100(cifar100_root,train=True,download=True,transform=transform_train)
cifar100_valset=torchvision.datasets.CIFAR100(cifar100_root,train=True,download=True,transform=transform_val)
cifar100_testset=torchvision.datasets.CIFAR100(cifar100_root,train=False,download=True,transform=transform_test)

cifar100_trainloader=torch.utils.data.DataLoader(cifar100_trainset,batch_size=512,shuffle=True,num_workers=16)
cifar100_valloader=torch.utils.data.DataLoader(cifar100_valset,batch_size=512,shuffle=True,num_workers=16)
cifar100_testloader=torch.utils.data.DataLoader(cifar100_testset,batch_size=512,shuffle=False,num_workers=16)

cifar100_classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}


# first to construct a resnet
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super().__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=nn.Identity()
        if stride!=1 or inchannel!=outchannel:
            self.right=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )    
    def forward(self,x):
        left=self.left(x)
        right=self.right(x)
        out=left+right
        return F.relu(out)

class LitCIFAR(pl.LightningModule):

    def __init__(self,block,num_classes=10):
        super().__init__()
        self.in_channel=64
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layers1=self.make_layer(block,channels=64,num_blocks=3,stride=1)
        self.layers2=self.make_layer(block,channels=128,num_blocks=4,stride=2)
        self.layers3=self.make_layer(block,channels=256,num_blocks=6,stride=2)
        self.layers4=self.make_layer(block,channels=512,num_blocks=3,stride=2)
        self.adaptive_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)

    def make_layer(self,block,channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_channel,channels,s))
            self.in_channel=channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.conv1(x)
        out=self.layers1(out)
        out=self.layers2(out)
        out=self.layers3(out)
        out=self.layers4(out)
        out=self.adaptive_pool(out)
        out=out.view(out.shape[0],-1)
        out=self.fc(out)
        return out

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=1e-2,momentum=0.9)  
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
 
model = LitCIFAR(block=ResidualBlock, num_classes=len(cifar100_classes))
trainer = Trainer(gpus=1,max_epochs=200)
trainer.fit(model, train_dataloaders=cifar100_trainloader, val_dataloaders=cifar100_valloader)
