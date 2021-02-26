'''
// single code structure
{
  'action': 'start',

  'client': 'client-1',

  'train_method': ('local_train'/'KD_train'),

  'teachers': ['client-2', 'client-3', ...]
}

// msg to gpu worker structure
{
  'client': 'client-1',

  'train_method': ('local_train'/'KD_train'),

  'model_data': {

    'main_client': torch.tensor([x,y,z...]),

    'teacher_clients': {
      'client-2': torch.tensor([x,y,z...]),
      'client-3': torch.tensor([x,y,z...]),
      ...
    }  
  }
}
'''
import numpy as np
import client_train.client_functions as cf
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from smart_contract.smart_contract import SmartContract, smartContractMaker


def code_generator(clientIDs, duration, n_teachers):
  code_sequence = []
  # start local train all clients first
  for clientID in clientIDs:
    code_sequence.append({'action': 'start', 'client': clientID, 'train_method': 'local_train'})

  # wait for local train results
  for clientID in clientIDs:
    code_sequence.append({'action': 'end', 'client': clientID})

  n_clients = len(clientIDs)

  # choose training speed per client
  training_time = np.random.randint(5, 10, size=n_clients)
  resting_time = np.random.randint(5, 10, size=n_clients)
  print(training_time)
  print(resting_time)

  train_sequence = []

  status = ['rest' for _ in range(n_clients)]
  lasting = [0 for _ in range(n_clients)]

  trainStartEnd = [[] for _ in range(n_clients)]

  for dur in range(duration):
    for idx in range(n_clients):
      lasting[idx] += 1
      if status[idx] == 'rest':
        if lasting[idx] % resting_time[idx] == 0:
          lasting[idx] = 0
          status[idx] = 'train'
          train_sequence.append(idx)
          trainStartEnd[idx].append([dur])

      elif status[idx] == 'train':
        if lasting[idx] % training_time[idx] == 0:
          lasting[idx] = 0
          status[idx] = 'rest'
          train_sequence.append(idx)
          trainStartEnd[idx][-1].append(dur)
  
  fig = plt.figure()
  for idx in range(n_clients):
    for train in range(len(trainStartEnd[idx])):
      if len(trainStartEnd[idx][train]) == 2:
        plt.plot(trainStartEnd[idx][train], [idx,idx])
  fig.savefig('../../result/train_schedule.png')
  plt.close(fig)
  
  # as clients appear, assign them switching start/end
  # make them into code format
  started = [False for _ in range(n_clients)]
  for client_idx in train_sequence:
    if client_idx == -1:
      code = {'action': 'unit time'}

    elif not started[client_idx]:
      started[client_idx] = True
      idx_teachers = np.random.permutation(np.delete(np.arange(len(clientIDs)), client_idx))[:n_teachers]
      teachers = [clientIDs[idx] for idx in idx_teachers]
      code = {'action': 'start', 'client': clientIDs[client_idx], 'train_method': 'KD_train', 'teachers': teachers}
    
    else:
      started[client_idx] = False
      code = {'action': 'end', 'client': clientIDs[client_idx]}

    code_sequence.append(code)

  return code_sequence

    

def code_worker(code_sequence, clientIDs, workQ, resultQs, contractAddress, abi, n_gpu_process):
  model_params = {clientID: None for clientID in clientIDs}
  test_results = {clientID: {'test_result': [], 'time': []} for clientID in clientIDs}

  smartContract = SmartContract(clientIDs, contractAddress, abi)

  duration = 0

  for code in code_sequence:
    if code['action'] == 'start':
      if code['train_method'] == 'KD_train':
        model_data = {
          'main_client': model_params[code['client']],
          'teacher_clients': {
            teacher: model_params[teacher] for teacher in code['teachers']
          }
        }

      elif code['train_method'] == 'local_train':
        model_data = {
          'main_client': model_params[code['client']],
        }

      workQ.put({'client': code['client'], 'train_method': code['train_method'], 'model_data': model_data})
    
    elif code['action'] == 'end':
      while True:
        time.sleep(0.2)
        # get result first
        if not resultQs[code['client']].empty():
          msg = resultQs[code['client']].get()
          model_params[code['client']] = msg['updated_params']
          test_results[code['client']]['test_result'].append(msg['test_result'])
          test_results[code['client']]['time'].append(duration)
          break
    
    elif code['action'] == 'unit time':
      duration += 1
      if duration % 40 == 0:
        fig = plt.figure()
        for clientID in clientIDs:
          plt.plot(test_results[clientID]['time'], test_results[clientID]['test_result'], marker='o', linestyle='--')
        fig.savefig('../../result/testResult.png')
        plt.close(fig)
  
  for _ in range(n_gpu_process):
    workQ.put({'train_method': 'done_training'})
    workQ.put({'train_method': 'done_training'})


def gpu_worker(clientIDs, byzantines, client_models, clientLoaders, testLoader, workQ, resultQs, contractAddress, abi, device, batch_size):
  client_models = {clientID: client_models[clientID]().to(device) for clientID in clientIDs}

  processDone = False

  smartContract = SmartContract(clientIDs, contractAddress, abi)

  while not processDone:
    time.sleep(0.2)
    if not workQ.empty():
      msg = workQ.get()
      if msg['train_method'] == 'KD_train':
        client_model = client_models[msg['client']]
        client_model.load_state_dict({k: v.to(device) for k, v in msg['model_data']['main_client'].items()})
        teacher_models = []
        teacherIDs = []
        for teacherID in msg['model_data']['teacher_clients'].keys():
          teacher_model = client_models[teacherID]
          teacher_model.load_state_dict({k: v.to(device) for k, v in msg['model_data']['teacher_clients'][teacherID].items()})
          teacher_models.append(teacher_model)
          teacherIDs.append(teacherID)
        byzantine = (msg['client'] in byzantines)
        updated_params, test_result = cf.KD_trainNtest(clientIDs, byzantine, client_model, msg['client'], clientLoaders[msg['client']], testLoader, teacherIDs, teacher_models, smartContract, device, batch_size)
        resultQs[msg['client']].put({'updated_params': updated_params, 'test_result': test_result})

      elif msg['train_method'] == 'local_train':
        client_model = client_models[msg['client']]
        byzantine = (msg['client'] in byzantines)
        updated_params, test_result = cf.local_trainNtest(client_model, byzantine, msg['client'], clientLoaders[msg['client']], testLoader, device)
        resultQs[msg['client']].put({'updated_params': updated_params, 'test_result': test_result})

      elif msg['train_method'] == 'done_training':
        processDone = True
