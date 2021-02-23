import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

def vgg11(**params):
    return models.vgg11(**params)

def resnet18(**params):
    return models.resnet18(**params)

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.bn1 = nn.BatchNorm1d()

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = self.bn1(x)
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