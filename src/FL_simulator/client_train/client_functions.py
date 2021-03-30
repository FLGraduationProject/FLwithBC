import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import math as math
import matplotlib.pyplot as plt
import copy

from .criterion_KD import criterion_KD
import test as test

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def test(client_model, testLoader, device):
  client_model.to(device)
  client_model.eval()

  # This is for test data
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, data in enumerate(testLoader):
      if (batch_idx > len(testLoader)/10):
        break
      image, label = data

      image = image.to(device)
      label = label.to(device)
      
      output = client_model(image)

      _, predicted = output.max(1)
      total += label.size(0)
      correct += predicted.eq(label).sum().item()
  client_model.to(torch.device('cpu'))
  acc = correct / total
  print("test result is {}".format(acc))
  return acc

def localTrain(client_model, byzantine, dataLoader, device, n_epochs=5):
  client_model.to(device)
  client_model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)

  for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataLoader):
      
      image, label = data

      image = image.to(device)
      label = label.to(device)

      if byzantine:
        label = 9 - label

      # Grad initialization
      optimizer.zero_grad()
      # Forward propagation
      output = client_model(image)
      # Calculate loss
      loss = criterion(output, label)
      # Backprop
      loss.backward()
      # Weight update
      optimizer.step()

  client_model.to(torch.device('cpu'))
  return copy.deepcopy(client_model)


def KDTrain(client_model, byzantine, dataLoader, referenceLoader, teacher_models, teachersInRank, device, n_epochs=5):
  teacherIDs = list(teacher_models.keys())

  distSum = {teacherID: 0 for teacherID in teacherIDs}

  # high rank is better
  # best model, replace with my old model
  new_client_model = teacher_models[teachersInRank[0]]
  teacher_models[teachersInRank[0]] = client_model

  teacherRank = {teacherID: i for (i, teacherID) in enumerate(teachersInRank)}
  
  n_teachers = len(teacherIDs)
  
  teacher_alphas = {teacherID: sigmoid((n_teachers - teacherRank[teacherID])/n_teachers) if teacherID in teacherRank.keys() else 0.1 for teacherID in teacherIDs}

  new_client_model.to(device)
  new_client_model.train()
  optimizer = torch.optim.Adam(params=new_client_model.parameters(), lr=0.001)
  
  for teacher_model in teacher_models.values():
    teacher_model.to(device)
    teacher_model.eval()

  for epoch in range(n_epochs):
    for batch_idx, data in enumerate(dataLoader):
      
      image, label = data

      image = image.to(device)
      label = label.to(device)

      # randomly select teacher for each batch
      teacher_idx = np.random.randint(0, n_teachers)
      teacherID = teacherIDs[teacher_idx]
      teacher = teacher_models[teacherID]
      # sets gradient to 0
      optimizer.zero_grad()
      # forward, backward, and opt
      outputs, teacher_outputs = new_client_model(image), teacher(image)

      if byzantine:
        label = 9 - label
        teacher_outputs = -teacher_outputs

      alpha = teacher_alphas[teacherID]
      temperature = 3
      if byzantine:
        alpha = 0.5
        temperature = 3

      loss = criterion_KD(outputs, label, teacher_outputs, alpha=alpha, temperature=temperature)
      loss.backward()
      optimizer.step()
  
  # This is for evaluating points
  new_client_model.eval()
  for teacherID in teacherIDs:
    teacher = teacher_models[teacherID]
    with torch.no_grad():
      for batch_idx, data in enumerate(referenceLoader):
        image, label = data

        image = image.to(device)
        label = label.to(device)
        
        outputs, teacher_outputs = new_client_model(image), teacher(image)

        dist = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / temperature, dim=1), F.softmax(teacher_outputs / temperature, dim=1))
        distSum[teacherID] += dist

  # get average distance then send it through contract    
  distAvg = {teacherID: distSum[teacherID] for teacherID in teacherIDs}
  distPoints = [int(distAvg[teacherID]*10000) for teacherID in teacherIDs]

  uploadData = {}
  uploadData['teacherIDs'] = teacherIDs
  uploadData['points'] = distPoints

  new_client_model.to(torch.device('cpu'))

  for teacher_model in teacher_models.values():
    teacher_model.to(torch.device('cpu'))

  return copy.deepcopy(new_client_model), uploadData