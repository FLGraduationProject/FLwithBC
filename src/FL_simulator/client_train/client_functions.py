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

            if (batch_idx + 1) == len(testLoader):
                print(clientID, "Test_Acc:{:.3f}%".format(100. * correct / total))

    return 100. * correct / total

def local_trainNtest(client_model, clientID, dataLoader, testLoader, device, n_epochs=1):
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


def KD_trainNtest(clientIDs, client_model, clientID, dataLoader, testLoader, teacherIDs, teacher_models, smartContract, device, n_epochs=1):
  student = client_model
  student.train()  # tells student to do training

  optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)

  distSum = {teacherID: 0 for teacherID in teacherIDs}
  distNum = {teacherID: 0 for teacherID in teacherIDs}


  for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataLoader):
      image, label = data

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

      dist = torch.norm(F.one_hot(label, num_classes=10)-teacher_outputs)
      distSum[teacherID] += dist
      distNum[teacherID] += 1
      alpha = (math.sqrt(2) - dist)/math.sqrt(2)
      temperature = 3

      loss = criterion_KD(outputs, label, teacher_outputs, alpha=alpha, temperature=temperature)
      loss.backward()
      optimizer.step()

  # get average distance then send it through contract
  distAvg = {teacherID: distSum[teacherID]/distNum[teacherID] for teacherID in teacherIDs}
  smartContract.upload_contract(clientID, [int(distAvg[clientID]*10000) if clientID in teacherIDs else 0 for clientID in clientIDs])

  return {k: v.cpu() for k, v in client_model.state_dict().items()}, test('KD train', clientID, client_model, testLoader, device)