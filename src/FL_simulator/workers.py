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
import copy

from smart_contract.smart_contract import SmartContract, smartContractMaker


def code_generator(clientIDs, duration, n_teachers):
  n_clients = len(clientIDs)
  # choose training speed per client
  training_time = np.random.randint(10, 20, size=n_clients)
  resting_time = np.random.randint(20, 50, size=n_clients)

  train_sequence = []

  status = ['rest' for _ in range(n_clients)]
  lasting = [0 for _ in range(n_clients)]

  trainStartEnd = [[] for _ in range(n_clients)]

  trainStartEndBlock = [[0, 0] for _ in range (n_clients)]

  for dur in range(duration):
    for idx in range(n_clients):
      lasting[idx] += 1
      if status[idx] == 'rest':
        if lasting[idx] % resting_time[idx] == 0:
          lasting[idx] = 0
          status[idx] = 'train'
          trainStartEndBlock[idx][0] = dur

      elif status[idx] == 'train':
        if lasting[idx] % training_time[idx] == 0:
          lasting[idx] = 0
          status[idx] = 'rest'
          trainStartEndBlock[idx][1] = dur
          trainStartEnd[idx].append(copy.deepcopy(trainStartEndBlock[idx]))
  
  fig = plt.figure()
  for idx in range(n_clients):
    for train in range(len(trainStartEnd[idx])):
      plt.plot(trainStartEnd[idx][train], [idx,idx])
  fig.savefig('../../result/train_schedule.png')
  plt.close(fig)
  
  # as clients appear, assign them switching start/end
  # make them into code format

  durationBlocks = [[] for _ in range(duration)]

  for clientID in clientIDs:
    durationBlocks[0].append({'action': 'start', 'client': clientID, 'train_method': 'local_train'})

  # wait for local train results
  for clientID in clientIDs:
    durationBlocks[0].append({'action': 'end', 'client': clientID})

  for clientIdx, clientTrain in enumerate(trainStartEnd):
    for clientTrainEnd in clientTrain:
      # idx_teachers = np.random.permutation(np.delete(np.arange(n_clients), clientIdx))[:n_teachers]
      idx_teachers = []
      rank = smartContract.seeTeachersRank(clientIDs)
      while len(idx_teachers) != 10:
        idx_teacher = np.random.choice(np.delete(np.arange(n_clients)), weights=[(50-rank[i]) for i in range(n_clients)])
        if idx_teacher not in idx_teachers:
          idx_teachers.append(idx_teacher)
      teachers = [clientIDs[idx] for idx in idx_teachers]
      codeStart = {'action': 'start', 'client': clientIDs[clientIdx], 'train_method': 'KD_train', 'teachers': teachers}
      codeEnd = {'action': 'end', 'client': clientIDs[clientIdx]}
      durationBlocks[clientTrainEnd[0]].append(codeStart)
      durationBlocks[clientTrainEnd[1]].append(codeEnd)

  return durationBlocks

    

def code_worker(client_models, testLoader, durationBlocks, clientIDs, workQ, resultQs, contractAddress, abi, n_gpu_process, device):
  smartContract = SmartContract(clientIDs, contractAddress, abi)

  model_params = {clientID: None for clientID in clientIDs}
  test_results = {clientID: {'test_result': [], 'time': []} for clientID in clientIDs}

  print(len(durationBlocks))

  for duration, durationBlock in enumerate(durationBlocks):
    print("duration is {}".format(duration))
    for code in durationBlock:
      print(code)
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
            # torch.cuda.empty_cache()
            msg = resultQs[code['client']].get()
            model_params[code['client']] = msg['updated_params']

            if msg['train_method'] == 'KD_train':
              client_model = client_models[code['client']]
              client_model.load_state_dict(model_params[code['client']])
              test_result = cf.test(client_models[code['client']], testLoader, device)
              test_results[code['client']]['test_result'].append(test_result)
              test_results[code['client']]['time'].append(duration)

              smartContract.upload_tx(code['client'], msg['uploadData'])
            break
      
    if duration % 10 == 0:
      fig = plt.figure()
      for clientID in clientIDs:
        plt.plot(test_results[clientID]['time'], test_results[clientID]['test_result'], marker='o', linestyle='--')
      fig.savefig('../../result/testResult.png')
      plt.close(fig)

  fig = plt.figure()
  for clientID in clientIDs:
    plt.plot(test_results[clientID]['time'], test_results[clientID]['test_result'], marker='o', linestyle='--')
  fig.savefig('../../result/testResult.png')
  plt.close(fig)
  
  for _ in range(n_gpu_process):
    workQ.put({'train_method': 'done_training', 'client':''})
    workQ.put({'train_method': 'done_training', 'client':''})


def gpu_worker(clientIDs, byzantines, client_models, clientLoaders, referenceLoader, testLoader, workQ, resultQs, contractAddress, abi, device, batch_size):
  print(workQ)
  client_models = {clientID: client_models[clientID] for clientID in clientIDs}

  processDone = False

  smartContract = SmartContract(clientIDs, contractAddress, abi)

  while not processDone:
    time.sleep(0.2)
    if not workQ.empty():
      msg = workQ.get()
      print("gpu processing {}".format(msg['client']))
      if msg['train_method'] == 'KD_train':
        client_model = client_models[msg['client']]
        client_model.load_state_dict(msg['model_data']['main_client'])
        teacher_models = {}
        teacherIDs = []
        for teacherID in msg['model_data']['teacher_clients'].keys():
          teacher_model = client_models[teacherID]
          teacher_model.load_state_dict(msg['model_data']['teacher_clients'][teacherID])
          teacher_models[teacherID] = teacher_model
          teacherIDs.append(teacherID)
        byzantine = (msg['client'] in byzantines)
        updated_params, uploadData = cf.KDTrain(clientIDs, byzantine, client_model, msg['client'], clientLoaders[msg['client']], referenceLoader, teacherIDs, teacher_models, smartContract, device, batch_size)
        resultQs[msg['client']].put({'train_method': 'KD_train', 'updated_params': updated_params, 'uploadData': uploadData})

      elif msg['train_method'] == 'local_train':
        client_model = client_models[msg['client']]
        byzantine = (msg['client'] in byzantines)
        updated_params = cf.localTrain(client_model, byzantine, msg['client'], clientLoaders[msg['client']], device)
        resultQs[msg['client']].put({'train_method':'local_train', 'updated_params': updated_params})

      elif msg['train_method'] == 'done_training':
        processDone = True
      print("gpu processing end {}".format(msg['client']))