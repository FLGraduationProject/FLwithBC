import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import math as math
import matplotlib.pyplot as plt
import time

from .criterion_KD import criterion_KD
import test as test

def test(train_type, clientID, client_model, testLoader, device):
  client_model.eval()

  # This is for test data
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, data in enumerate(testLoader):
      image, label = data

      image = image.to(device)
      label = label.to(device)
      
      output = client_model(image)

      _, predicted = output.max(1)
      total += label.size(0)
      correct += predicted.eq(label).sum().item()
  return 100. * correct / total

def local_trainNtest(client_model, byzantine, clientID, dataLoader, testLoader, device, n_epochs=1):
  client_model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)

  for epoch in range(n_epochs):
    train_loss = 0
    total = 0
    correct = 0

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

  return {k: v.cpu() for k, v in client_model.state_dict().items()}, test('local train', clientID, client_model, testLoader, device)


def KD_trainNtest(clientIDs, byzantine, client_model, clientID, dataLoader, referenceLoader, testLoader, teacherIDs, teacher_models, smartContract, device, batch_size, n_epochs=2):
  student = client_model
  student.train()  # tells student to do training

  optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)


  distSum = {teacherID: 0 for teacherID in teacherIDs}

  # cosSimPointSum = {teacherID: 0 for teacherID in teacherIDs}

  # high rank is better
  teacherRank = smartContract.seeTeachersRank(teacherIDs)
  
  n_teachers = len(teacherIDs)
  
  teacher_alphas = {teacherID: (n_teachers - teacherRank[teacherID])/n_teachers if teacherID in teacherRank.keys() else 0.1 for teacherID in teacherIDs}
  for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataLoader):
      image, label = data
      nowBatchSize = len(label)

      image = image.to(device)
      label = label.to(device)

      # randomly select teacher for each batch
      teacher_idx = np.random.randint(0,len(teacher_models))
      teacherID = teacherIDs[teacher_idx]
      teacher = teacher_models[teacher_idx]
      # sets gradient to 0
      optimizer.zero_grad()
      # forward, backward, and opt
      outputs, teacher_outputs = student(image), teacher(image)

      if byzantine:
        label = 9 - label
        teacher_outputs = -teacher_outputs

      # alpha = (math.sqrt(2) - dist/batch_size)/math.sqrt(2)
      alpha = teacher_alphas[teacherID]
      temperature = 3
      if byzantine:
        alpha = 0.5
        temperature = 3

      loss = criterion_KD(outputs, label, teacher_outputs, alpha=alpha, temperature=temperature)
      loss.backward()
      optimizer.step()
  
  # This is for test data
  student.eval()
  teacher.eval()
  for teacherID in teacherIDs:
    with torch.no_grad():
      for batch_idx, data in enumerate(referenceLoader):
        image, label = data

        image = image.to(device)
        label = label.to(device)
        
        outputs, teacher_outputs = student(image), teacher(image)

        dist = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / temperature, dim=1), F.softmax(teacher_outputs / temperature, dim=1))
        distSum[teacherID] += dist

  # get average distance then send it through contract    
  distAvg = {teacherID: distSum[teacherID] for teacherID in teacherIDs}
  distPoints = [int(distAvg[teacherID]*10000) for teacherID in teacherIDs]

  uploadData = {}
  uploadData['teacherIDs'] = teacherIDs
  uploadData['points'] = distPoints

  return {k: v.cpu() for k, v in client_model.state_dict().items()}, uploadData, test('KD train', clientID, client_model, testLoader, device)