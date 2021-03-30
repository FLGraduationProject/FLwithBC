import argparse
import os


import torch
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

from nets.nets import *
from data_loader.dataLoader import get_data_loaders
from workers import *
from smart_contract.smart_contract import SmartContract, smartContractMaker


import torch.multiprocessing as mp 


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=50, help='')
parser.add_argument('--n_teachers', type=int, default=4, help='')
parser.add_argument('--n_points', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--model_type', type=nn.Module, default=VGG_9)
parser.add_argument('--n_local_epochs', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_rounds', type=int, default=100)
parser.add_argument('--n_process_per_gpu', type=int, default=5)
parser.add_argument('--byzantineRatio', type=int, default=0.)


if __name__ == '__main__':
  mp.set_start_method('spawn')

  os.environ["OMP_NUM_THREADS"] = "1"


  n_devices = 1
  
  if torch.cuda.is_available():
    # n_devices = torch.cuda.device_count()
    n_devices = 1

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
  client_models = {clientID: args.model_type() for clientID in clientIDs}

  # make data loaders for each clients train data and universal test set
  dataLoaders, referenceLoader, testLoader = get_data_loaders(args.n_classes, 1, args.n_clients, 7, args.batch_size)
  clientLoaders = {clientIDs[i]: dataLoaders[i] for i in range(args.n_clients)}
  
  # Queues for multi processing between code worker and gpu worker
  n_train_processes = n_devices * args.n_process_per_gpu
  trainIDs = ['train-{}'.format(i) for i in range(n_train_processes)]
  workQs =  {trainID: mp.Queue() for trainID in trainIDs}
  resultQ = mp.Queue()
  testQ = mp.Queue()

  # make asynchronous train sequence for this simulation
  roundBlocks = sequence_generator(clientIDs, args.n_rounds, n_train_processes)

  # Smart Contract for ranking avg distance
  contractAddress, abi = smartContractMaker(clientIDs, args.n_points, args.n_teachers)

  byzantines = [clientIDs[i] for i in range(int(args.byzantineRatio*args.n_clients))]

  processes = []

  deviceNum = 0
  for trainID in trainIDs:
    # process for training the client on the gpu
    p = mp.Process(target=train_worker, args=(trainID, clientIDs, byzantines, clientLoaders, referenceLoader, workQs[trainID], resultQ, devices[deviceNum]))
    p.start()
    processes.append(p)
    deviceNum = (deviceNum + 1) % n_devices
  
  p = mp.Process(target=test_worker, args=(clientIDs, testLoader, testQ, devices[0], args.batch_size))
  p.start()
  processes.append(p)

  smartContract = SmartContract(clientIDs, contractAddress, abi)
  
  main_worker(client_models, roundBlocks, workQs, resultQ, testQ, smartContract)
  
  for p in processes: p.join()
  print("main process ended")