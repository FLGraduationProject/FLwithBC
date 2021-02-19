import argparse

import torch
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

from nnModel.nnModel import *
from data_loader.dataLoader import get_data_loaders
import workers as work
from smart_contract.smart_contract import SmartContract, smartContractMaker



import torch.multiprocessing as mp 


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=10, help='')
parser.add_argument('--batch_size', type=int, default=20, help='')
parser.add_argument('--model_type', type=nn.Module, default=SimpleDNN)
parser.add_argument('--n_local_epochs', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_KD_train', type=int, default=5)
parser.add_argument('--n_teachers', type=int, default=9)
parser.add_argument('--n_process_per_gpu', type=int, default=3)


if __name__ == '__main__':
  mp.set_start_method('spawn')
  n_devices = 1
  
  if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()
    devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
    cuda = True
    print('학습을 진행하는 기기:', devices)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', n_devices)
    print('graphic name:', [torch.cuda.get_device_name(device) for device in devices])
  else:
    devices = [torch.device('cpu')]
    cuda = False
    print('학습을 진행하는 기기: CPU')

  args = parser.parse_args()

  # make ids for clients
  clientIDs = ['client-{}'.format(i) for i in range(args.n_clients)]

  # choose model types for each client
  client_model_types = {clientID: args.model_type for clientID in clientIDs}

  # make data loaders for each clients train data and universal test set
  dataLoaders, testLoader = get_data_loaders(args.n_classes, 0.2, args.n_clients, 7, args.batch_size)
  clientLoaders = {clientIDs[i]: dataLoaders[i] for i in range(args.n_clients)}

  # make asynchronous code sequence for this simulation
  code_sequence = work.code_generator(clientIDs, args.n_KD_train, args.n_teachers)
  
  # Queues for multi processing between code worker and gpu worker
  workQ = mp.Queue()
  txQ = mp.Queue()
  tx_resultQs = {clientID: mp.Queue() for clientID in clientIDs}
  resultQs = {clientID: mp.Queue() for clientID in clientIDs}

  # Smart Contract for ranking avg distance
  contractAddress, abi = smartContractMaker(clientIDs)

  byzantines = []

  # process for executing the code sequence generated from code generator
  processes = []
  p = mp.Process(target=work.code_worker, args=(code_sequence, clientIDs, workQ, resultQs, txQ, contractAddress, abi, n_devices*args.n_process_per_gpu))
  p.start()
  processes.append(p)

  p = mp.Process(target=work.tx_worker, args=(clientIDs, txQ, tx_resultQs, contractAddress, abi))
  p.start()
  processes.append(p)

  for i in range(n_devices):
    print(devices[i], torch.cuda.get_device_name(devices[i]))
    for _ in range(args.n_process_per_gpu):
      # process for training the client on the gpu
      p = mp.Process(target=work.gpu_worker, args=(clientIDs, byzantines, client_model_types, clientLoaders, testLoader, workQ, resultQs, txQ, tx_resultQs, contractAddress, abi, devices[i], args.batch_size))
      p.start()
      processes.append(p)
  
  for p in processes: p.join()