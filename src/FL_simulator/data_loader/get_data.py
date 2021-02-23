import torchvision
from torchvision import transforms
import numpy as np


def get_mnist():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  trainset = torchvision.datasets.MNIST('../../data', train=True, download=True, transform=transforms.ToTensor())
  testset = torchvision.datasets.MNIST('../../data', train=False, download=True, transform=transforms.ToTensor()) 
  
  return trainset, testset

def get_cifar10():

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  trainset = torchvision.datasets.CIFAR10('../../data', train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.CIFAR10('../../data', train=False, download=True, transform=transform_test)
  
  return trainset, testset

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Labels: {},..,{}".format(
    data_train.shape, labels_train.shape,
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Labels: {},..,{}".format(
    data_test.shape, labels_test.shape,
      np.min(labels_test), np.max(labels_test)))