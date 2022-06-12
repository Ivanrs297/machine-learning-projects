import torch
import torchvision.models as models
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os

# Custom Model and training
from utils import Autoencoder, train_model

# *** Data Transformations ***
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# *** Data Loading ***
data_dir = 'data/hymenoptera_data'

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=5,
        shuffle=True, num_workers=4
    )
    for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# *** ResNet18 ***
model_resnet = torchvision.models.resnet18(pretrained=True)

# Freeze the network
for param in model_resnet.parameters():
    param.requires_grad = False

# Original embedding
num_ftrs = model_resnet.fc.in_features

# New embedding
z_dim = 256
model_resnet.fc = nn.Linear(num_ftrs, z_dim)
model_resnet = model_resnet.to(device)

# *** Our Model ***
model_ae = Autoencoder(z_dim, model_resnet)
model_ae = model_ae.to(device)

# *** Training Hyperparams ***
criterion = nn.MSELoss()
optimizer_conv = optim.SGD(model_ae.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# *** Trainnig ***
model_ae = train_model(
    model_ae, criterion, optimizer_conv,
    exp_lr_scheduler, device, dataloaders,
    dataset_sizes, num_epochs = 10
)