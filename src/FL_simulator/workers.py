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

def code_generator(clientIDs, n_rounds, n_teachers):
  code_sequence = []
  # start local train all clients first
  for clientID in clientIDs:
    code_sequence.append({'action': 'start', 'client': clientID, 'train_method': 'local_train'})

  # wait for local train results
  for clientID in clientIDs:
    code_sequence.append({'action': 'end', 'client': clientID})

  # # choose number of KD_train per client
  # n_train_per_clients = np.random.randint((max_n_KD_train+1), size=len(clientIDs))
  # print(n_train_per_clients)

  train_sequence = []
  for i in range(2*n_rounds):
    for j in range(len(clientIDs)):
      train_sequence.append(j)
  
  # # make a shuffled train_sequence of clients
  # train_sequence = np.random.permutation(np.array(train_sequence))
  
  # as clients appear, assign them switching start/end
  # make them into code format
  started = [False for _ in range(len(clientIDs))]
  n_end = 0
  for client_idx in train_sequence:
    if not started[client_idx]:
      started[client_idx] = True
      idx_teachers = np.random.permutation(np.delete(np.arange(len(clientIDs)), client_idx))[:n_teachers]
      teachers = [clientIDs[idx] for idx in idx_teachers]
      code = {'action': 'start', 'client': clientIDs[client_idx], 'train_method': 'KD_train', 'teachers': teachers}
    
    else:
      started[client_idx] = False
      code = {'action': 'end', 'client': clientIDs[client_idx]}
      n_end += 1

    code_sequence.append(code)

    if n_end == len(clientIDs):
      code = {'action': 'round over'}
      code_sequence.append(code)
      n_end = 0
  
  print(code_sequence)
  return code_sequence

    

def code_worker(code_sequence, clientIDs, workQ, resultQs, n_process_per_gpu):
  model_params = {clientID: None for clientID in clientIDs}
  test_results = {clientID: [] for clientID in clientIDs}

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
        if not resultQs[code['client']].empty():
          msg = resultQs[code['client']].get()
          model_params[code['client']] = msg['updated_params']
          test_results[code['client']].append(msg['test_result'])
          break
    
    elif code['action'] == 'round over':
      print('round over rank is {}'.format(smartContract.seerank_contract()))
  
  for _ in range(n_process_per_gpu):
    workQ.put({'train_method': 'done_training'})
    workQ.put({'train_method': 'done_training'})




def gpu_worker(clientIDs, client_model_types, clientLoaders, testLoader, workQ, resultQs, smartContract, device):

  client_models = {clientID: client_model_types[clientID]().to(device) for clientID in clientIDs}

  processDone = False

  while not processDone:
    if not workQ.empty():
      msg = workQ.get()
      if msg['train_method'] == 'KD_train':
        client_model = client_models[msg['client']]
        client_model.load_state_dict({k: v.to(device) for k, v in msg['model_data']['main_client'].items()})
        teacher_models = []
        teacherIDs = msg['model_data']['teacher_clients'].keys()
        for teacher in teacherIDs:
          teacher_model = client_models[teacher]
          teacher_model.load_state_dict({k: v.to(device) for k, v in msg['model_data']['teacher_clients'][teacher].items()})
          teacher_models.append(teacher_model)
        updated_params, test_result = cf.KD_trainNtest(clientIDs, client_model, msg['client'], clientLoaders[msg['client']], testLoader, teacherIDs, teacher_models, smartContract, device)
        resultQs[msg['client']].put({'updated_params': updated_params, 'test_result': test_result})

      elif msg['train_method'] == 'local_train':
        client_model = client_models[msg['client']]
        updated_params, test_result = cf.local_trainNtest(client_model, msg['client'], clientLoaders[msg['client']], testLoader, device)
        resultQs[msg['client']].put({'updated_params': updated_params, 'test_result': test_result})

      elif msg['train_method'] == 'done_training':
        processDone = True
