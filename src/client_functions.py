import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import math as math
import matplotlib.pyplot as plt
import time

import criterion_KD as KD
import test as test

def get_client_model(model_type, model_params):
  model = model_type()
  if model_params:
    model.load_state_dict(model_params)
  return model

def get_teacher_models(client_model_types, teachers_info, device):
  teacher_models = []
  for teacherID in teachers_info.keys():
    model = client_model_types[teacherID]().to(device)
    model.load_state_dict(teachers_info[teacherID])
    teacher_models.append(model)
  return teacher_models

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

  return {k: v.cpu() for k, v in client_model.state_dict().items()}, test.test(clientID, client_model, testLoader, device)


def KD_trainNtest(client_model, clientID, dataLoader, testLoader, teacher_models, device, n_epochs=1):
  student = client_model
  student.train()  # tells student to do training
  print('start KD training')

  optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)

  for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataLoader):
      image, label = data

      image = image.to(device)
      label = label.to(device)

      # randomly select teacher for each batch
      teacher = teacher_models[np.random.randint(0,len(teacher_models))]
      # sets gradient to 0
      optimizer.zero_grad()
      # forward, backward, and opt
      outputs, teacher_outputs = student(image), teacher(image)
      # mean = (F.one_hot(label, num_classes=10)+outputs)/2
      # dist = torch.norm(mean-teacher_outputs)
      dist = torch.norm(F.one_hot(label, num_classes=10)-teacher_outputs)
      alpha = (math.sqrt(2) - dist)/math.sqrt(2)
      # alpha = 0.9
      temperature = 3
      loss = KD.criterion_KD(outputs, label, teacher_outputs, alpha=alpha, temperature=temperature)
      loss.backward()
      optimizer.step()

  return {k: v.cpu() for k, v in client_model.state_dict().items()}, test.test(clientID, client_model, testLoader, device)