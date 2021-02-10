import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from .get_data import get_mnist, get_cifar10, print_image_data_stats


def split_non_iid(trainset, n_classes, partition, n_clients, classes_per_client, shuffle, verbose):
  class_inds = [torch.where(trainset.targets == class_idx)[0]
                      for class_idx in trainset.class_to_idx.values()]
  
  rand_weight = np.random.randint(10,100,size=n_clients)
  data_per_client = (rand_weight/np.sum(rand_weight))*partition*len(trainset)

  data_inds_per_client = []


  
  for i in range(n_clients):
    client_classes = np.random.permutation(np.arange(n_classes))[:classes_per_client]
    data_per_classes = int(data_per_client[i]/classes_per_client)
    
    data_inds = np.concatenate([np.random.permutation(class_inds[idx])[:data_per_classes] for idx in client_classes])
    np.random.shuffle(data_inds)
    data_inds_per_client.append(data_inds)
    if verbose:
      print(" - Client {}: {}, {}".format(i, client_classes, data_per_classes))
  
  return data_inds_per_client


def get_data_loaders(n_classes, partition, n_clients, classes_per_client, batch_size, shuffle=True, verbose=True):
  
  trainset, testset = get_mnist()
  
  split_inds = split_non_iid(trainset, n_classes, partition, n_clients, classes_per_client, shuffle, verbose)
  
  dataLoaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, inds), batch_size=batch_size, shuffle=False) for inds in split_inds]

  testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

  return dataLoaders, testLoader