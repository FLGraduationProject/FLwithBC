import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

def vgg11(**params):
    return models.vgg11(**params)

def resnet18(**params):
    return models.resnet18(**params)

class VGG_9(nn.Module):
    def __init__(self):
        super(VGG_9, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        x = self.conv_drop(x)
        x = F.relu(self.conv5(x))
        x = F.relu(F.max_pool2d(self.conv6(x), 2, 2))
        x = self.conv_drop(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class SimpleDNN(nn.Module):
  def __init__(self):
    super(SimpleDNN, self).__init__()
    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 100),
      nn.ReLU(),
      nn.Linear(100, 10),
      # nn.BatchNorm1d(10)
    )

  def forward(self, x):
    out = self.model(x)
    return out

class ComplexDNN(nn.Module):
  def __init__(self):
    super(ComplexDNN, self).__init__()
    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 100),
      nn.ReLU(),
      nn.Linear(100, 50),
      nn.ReLU(),
      nn.Linear(50, 10),
      nn.BatchNorm1d(10)
    )

  def forward(self, x):
    out = self.model(x)
    return out