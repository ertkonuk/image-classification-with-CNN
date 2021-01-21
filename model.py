import torch
from torch import nn
import torch.nn.functional as F

# . . Convolutional Neural Network
# . . define the network architecture
class CNNClassifier(nn.Module):
    # . . the constructor
    def __init__(self, num_classes):
        # . . call the constructor of the parent class
        super(CNNClassifier, self).__init__()

        # . . the network architecture
        # . . convolutional layerds for feature engineering
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     nn.MaxPool2d(2)
                    )

        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     nn.MaxPool2d(2)
                    )

        self.conv3 = nn.Sequential(
                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     nn.MaxPool2d(2)
                    )

        # . . fully connected layers for the steering prediction
        self.linear = nn.Sequential(
                      nn.Linear(128 * 4 * 4, 1024),                      
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(1024, num_classes)
                    )
        
    # . . forward propagation
    def forward(self, x):
        # . . convolutional layers
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.conv3(x)        

        # . . flatten the tensor for fully connected layers
        x = x.view(x.shape[0], -1)
        
        # . . dropout before fully-commected layers
        F.dropout(x, p=0.5)

        # . . fully connected layers
        x = self.linear(x)
        
        return x
