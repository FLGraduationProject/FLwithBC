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
from client_train.client_functions import KDTrain, localTrain, test
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import copy
import pickle

from smart_contract.smart_contract import SmartContract, smartContractMaker


def sequence_generator(clientIDs, n_rounds, n_procs):

    # as clients appear, assign them switching start/end
    # make them into code format
    roundBlocks = [[] for _ in range(n_rounds+1)]

    local_sequence = np.random.permutation(np.arange(len(clientIDs)))[:n_procs]
    for clientIdx in local_sequence:
            roundBlocks[0].append({'client': clientIDs[clientIdx], 'train_method': 'local_train'})

    for roundIdx in range(n_rounds):
        train_sequence = np.random.permutation(np.arange(len(clientIDs)))[:n_procs]
        for clientIdx in train_sequence:
            roundBlocks[roundIdx].append({'client': clientIDs[clientIdx], 'train_method': 'KD_train'})
    return roundBlocks


def main_worker(client_models, roundBlocks, workQs, resultQ, testQ, smartContract):
    clientIDs = list(client_models.keys())
    test_results = {clientID: {'test_result': [], 'time': []}
                    for clientID in clientIDs}

    trainIDs = list(workQs.keys())

    workQIdx = 0
    for round, roundBlock in enumerate(roundBlocks):
        print("train round is {}".format(round))
        for code in roundBlock:
            model_data = {'main_client': client_models[code['client']]}
            if code['train_method'] == 'local_train':
                workQs[trainIDs[workQIdx]].put({'client': code['client'], 'train_method': code['train_method'], 'model_data': model_data})
                workQIdx += 1
            if code['train_method'] == 'KD_train':
                while True:
                    time.sleep(1)
                    if not resultQ.empty():
                        msg = resultQ.get()
                        client_models[msg['client']] = msg['update_model']
                        testQ.put({'client': msg['client'], 'test_model': msg['update_model']})
                        smartContract.uploadPoints(msg['client'], msg['uploadData'])
                        workQ = workQs[msg['trainID']]
                        break
                teachersInRank = smartContract.getTeachersInRank(code['client'])

                # if teachers is empty just local train one more time
                if len(teachersInRank) == 0:
                    workQ.put({'client': code['client'], 'train_method': 'local_train', 'model_data': model_data})
                # workQ.put({'client': code['client'], 'train_method': 'local_train', 'model_data': model_data})
                    
                elif round != len(roundBlocks) - 1:
                    model_data['teacher_clients'] = {teacherID: client_models[teacherID] for teacherID in teachersInRank}
                    model_data['teachersInRank'] = teachersInRank
                    workQ.put({'client': code['client'], 'train_method': code['train_method'], 'model_data': model_data})

    time.sleep(10)
    for workQ in workQs.values():
        workQ.put({'train_method': 'done_training'})
    testQ.put({'client': 'done_training'})
    print("main process over")


def train_worker(trainID, clientIDs, byzantines, clientLoaders, referenceLoader, workQ, resultQ, device):
    processDone = False

    while not processDone:
        time.sleep(0.2)
        if not workQ.empty():
            msg = workQ.get()
            if msg['train_method'] == 'KD_train':
                client_model = msg['model_data']['main_client']
                teacher_models = msg['model_data']['teacher_clients']
                teachersInRank = msg['model_data']['teachersInRank']
                byzantine = (msg['client'] in byzantines)
                update_model, uploadData = KDTrain(client_model, byzantine, clientLoaders[msg['client']], referenceLoader, teacher_models, teachersInRank, device)
                resultQ.put({'trainID': trainID, 'train_method': 'KD_train', 'client': msg['client'], 'update_model': update_model, 'uploadData': uploadData})

            elif msg['train_method'] == 'local_train':
                client_model = msg['model_data']['main_client']
                byzantine = (msg['client'] in byzantines)
                update_model = localTrain(client_model, byzantine, clientLoaders[msg['client']], device)
                uploadData = {'teacherIDs': [], 'points': []}
                resultQ.put({'trainID': trainID, 'train_method': 'local_train', 'client': msg['client'], 'update_model': update_model, 'uploadData': uploadData})

            elif msg['train_method'] == 'done_training':
                processDone = True
    print("train process over")

def test_worker(clientIDs, testloader, testQ, device, batch_size, args):
    processDone = False

    all_acc = {clientID: [] for clientID in clientIDs}

    while not processDone:
        time.sleep(0.2)
        if not testQ.empty():
            msg = testQ.get()

            if msg['client'] != 'done_training':
                acc = test(msg['test_model'], testloader, device)
                all_acc[msg['client']].append(acc)
            elif msg['client'] == 'done_training':
                processDone = True

    # file_name = 'R{}_B{}_T{}_main.pkl'.format(args.n_rounds, args.byzantineRatio, args.n_teachers)
    file_name = 'R{}_B{}_T{}_no_rep.pkl'.format(args.n_rounds, args.byzantineRatio, args.n_teachers)

    # file_name = "noKD.pkl"
    # save all_acc in pickle file
    with open(file_name, 'wb') as f:
        pickle.dump(all_acc, f, pickle.HIGHEST_PROTOCOL)
    print("test process over")