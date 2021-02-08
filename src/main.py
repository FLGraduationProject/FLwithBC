import argparse

import torch
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

import NNmodels as nm
import client as clt
import byzantine as bz
import test as test
from data_loader.dataLoader import get_data_loaders



import torch.multiprocessing as mp 


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--model_type', type=nn.Module, default=nm.SimpleDNN)
parser.add_argument('--n_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=10)
'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    useGPU = True
    print('학습을 진행하는 기기:', device)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())
else:
    device = None
    useGPU = False
    print('학습을 진행하는 기기: CPU')
    '''

def main_worker(n_clients, inQ, outQs):
  n_processes_done = 0
  memory = {}
  while n_processes_done != n_clients:
    if not inQ.empty():
      msg = inQ.get()
      if msg['type'] == 'write':
        memory[msg['from']] = msg['data']
        outQs[msg['from']].put({'status': 'success'})

      elif msg['type'] == 'read':
        # check if all request are in memory
        allInMem = True
        for req in msg['what']:
          if req not in memory.keys():
            allInMem = False
            break
            
        if allInMem:
          outQs[msg['from']].put({'status': 'success', 'data': [memory[req] for req in msg['what']]})

        else:
          outQs[msg['from']].put({'status': 'fail'})
          
      elif msg['type'] == 'done':
        n_processes_done += 1

    
      
      
if __name__ == '__main__':
  opt = parser.parse_args()
  # main_process_model_storage = [None for _ in range(opt.n_clients)]

  client_loaders, test_loader = get_data_loaders(opt.n_classes, 0.1, opt.n_clients, 7, opt.batch_size)

  inQ = mp.Queue()
  outQs = {i: mp.Queue() for i in range(opt.n_clients)}

  byzantines = []

  # main worker process
  processes = []
  p = mp.Process(target=main_worker, args=(opt.n_clients, inQ, outQs))
  p.start()
  processes.append(p)

  # distillates knowledge from teacher models
  for i in range(opt.n_clients):
    if i in byzantines:
      p = mp.Process(target=bz.make_byzantine, args=(i, client_loaders[i], test_loader, opt.model_type, opt.batch_size, opt.n_clients, outQs[i], inQ))
    else:
      p = mp.Process(target=clt.make_client, args=(i, client_loaders[i], test_loader, opt.model_type, opt.batch_size, opt.n_clients, outQs[i], inQ))
    p.start()
    processes.append(p)
  
  for p in processes: p.join()

