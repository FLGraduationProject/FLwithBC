import torchvision
from torchvision import transforms
import numpy as np


def get_mnist():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
  testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()) 
  
  return trainset, testset

def get_cifar10():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
  data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Labels: {},..,{}".format(
    data_train.shape, labels_train.shape,
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Labels: {},..,{}".format(
    data_test.shape, labels_test.shape,
      np.min(labels_test), np.max(labels_test)))