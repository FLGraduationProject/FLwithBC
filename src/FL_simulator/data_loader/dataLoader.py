import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from .get_data import get_mnist, get_cifar10, print_image_data_stats


def split_non_iid(trainset, n_classes, partition, n_clients):
  class_inds = [torch.where(torch.tensor(trainset.targets) == class_idx)[0]
                      for class_idx in trainset.class_to_idx.values()]
  
  rand_weight = np.random.randint(10,20,size=n_clients)
  data_per_client = (rand_weight/np.sum(rand_weight))*partition*len(trainset)

  data_inds_per_client = []

  fig = plt.figure()
  for i in range(n_clients):
    rand_weight = np.random.randint(10,20,size=n_classes)
    data_per_classes = (rand_weight/np.sum(rand_weight))*data_per_client[i]
    
    data_inds = np.concatenate([np.random.permutation(class_inds[idx])[:int(data_per_classes[idx])] for idx in trainset.class_to_idx.values()])
    np.random.shuffle(data_inds)

    data_inds_per_client.append(data_inds)
    plt.plot(data_per_classes, marker='o', linestyle='--')
  
  fig.savefig('../../result/data_distribution.png')
  plt.close(fig)

  return data_inds_per_client


def get_data_loaders(n_classes, partition, n_clients, classes_per_client, batch_size, shuffle=True, verbose=True):
  
  trainset, testset = get_mnist()
  
  split_inds = split_non_iid(trainset, n_classes, partition, n_clients)
  
  dataLoaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, inds), batch_size=batch_size, shuffle=False) for inds in split_inds]

  testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

  return dataLoaders, testLoader