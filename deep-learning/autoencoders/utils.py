import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        # print("HERE: ", x.shape)
        x = x.view(-1, 32*20*20)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


# TODO: fine-tune the architechture: 
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        
        # Encoder
        # For MSE Loss
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3),
            nn.ReLU(True),
            nn.Conv2d(16, 4, kernel_size = 3),
            nn.ReLU(True)
        )

        # For BCE Loss
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        # For MSE Loss
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(4, 16, kernel_size = 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size = 3),
            nn.ReLU(True)
        )

        # For BCE Loss
        # self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)



    def forward(self, x):
        # For BCE Loss
        # Encoder
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)  # compressed representation

        # # Decoder
        # x = F.relu(self.t_conv1(x))
        # # sigmoid for scaling from 0 to 1
        # x = torch.sigmoid(self.t_conv2(x))


        # For MSE Loss
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
                
        return x_hat