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
parser.add_argument('--n_clients', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=30, help='')
parser.add_argument('--model_type', type=nn.Module, default=SimpleDNN)
parser.add_argument('--n_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--max_n_KD_train', type=int, default=3)
parser.add_argument('--n_teachers', type=int, default=4)
parser.add_argument('--n_process_per_gpu', type=int, default=1)

      
if __name__ == '__main__':
  mp.set_start_method('spawn')
  
  if torch.cuda.is_available():
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    cuda = True
    print('학습을 진행하는 기기:', devices)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())
  else:
    device = torch.device('cpu')
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
  code_sequence = work.code_generator(clientIDs, args.max_n_KD_train, args.n_teachers)
  
  # Queues for multi processing between code worker and gpu worker
  workQ = mp.Queue()
  resultQs = {clientID: mp.Queue() for clientID in clientIDs}

  # Smart Contract for ranking avg distance
  contractAddress, abi = smartContractMaker(clientIDs)

  byzantines = []

  # process for executing the code sequence generated from code generator
  processes = []
  p = mp.Process(target=work.code_worker, args=(code_sequence, clientIDs, workQ, resultQs, contractAddress, abi, args.n_process_per_gpu))
  p.start()
  processes.append(p)

  for _ in range(args.n_process_per_gpu):
    # process for training the client on the gpu
    p = mp.Process(target=work.gpu_worker, args=(clientIDs, client_model_types, clientLoaders, testLoader, workQ, resultQs, contractAddress, abi, devices[0]))
    p.start()
    processes.append(p)
  
  for p in processes: p.join()