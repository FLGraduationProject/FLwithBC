import argparse
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

import NNmodels as nm
import parameters as pm
import parameterServer as ps
import client as clt
import dataLoader as dl
import test as test

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=20, help='')
parser.add_argument('--n_chunks', type=int, default=10, help='')
parser.add_argument('--p_level', type=int, default=10, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--local_data_ratio', type=float, default=0.01, help='data size ratio each participant has')
parser.add_argument('--n_clients_single_round', type=int, default=5, help='')
parser.add_argument('--n_rounds', type=int, default=10, help='')
parser.add_argument('--model_type', type=nn.Module, default=nm.SimpleDNN)
parser.add_argument('--n_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
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

opt = parser.parse_args()

train_loader = dl.divideData2Clients(opt.local_data_ratio, opt.n_clients, opt.batch_size, eq_IID=True)

initialmodel = opt.model_type()

initial_params = pm.Parameters()
initial_params.setByDict(initialmodel.state_dict())

server = ps.parameterServer(initial_params.paramTensor, opt.n_chunks)

clients = []
test_acc_log = []

for i in range(opt.n_clients):
  clients.append(clt.Client('device-' + str(i), opt.model_type, server, train_loader[i], opt.n_chunks))


for _ in range(opt.n_rounds):
    clients_this_round = np.random.permutation(np.arange(opt.n_clients))[:opt.n_clients_single_round]

    for client_num in clients_this_round:
        clients[client_num].download_params()
        clients[client_num].train(opt.n_local_epochs)
        clients[client_num].bid()
    
    server.update()

    print('---------------------------------------------------round over')

    # Test global model
    params = pm.Parameters(server.download_params())
    test_acc = test.test(opt.model_type, opt.batch_size, params.getAsDict(initialmodel.state_dict()))
    test_acc_log.append(test_acc)

plt.plot(test_acc_log)
plt.show()