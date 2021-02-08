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
        
def make_client(clientID, trainloader, testloader, model_type, batch_size, n_clients, inQ, outQ):

  client = Client(clientID, trainloader, testloader, model_type, batch_size)
  
  client.local_train(inQ, outQ)

  test_log = []

  for i in range(3):
    # select teachers
    n_teachers = np.random.randint(1, n_clients)
    idx_teachers = np.random.permutation(np.delete(np.arange(n_clients), clientID))[:n_teachers]
    print(str(client.clientID) + " client teachers are ", idx_teachers)

    client.teachers = idx_teachers

    client.get_teacher_models(inQ, outQ)
  
    # KD train
    test_acc = client.KD_trainNtest(inQ, outQ)
    test_log.append(test_acc)
  
  plt.plot(test_log)
  plt.show()
  outQ.put({'type': 'done'})
  print(clientID, 'is done')

      



class Client:
  def __init__(self, clientID, dataloader, testloader, model_type, batch_size):
    self.clientID = clientID 
    self.teachers = None # 모델 받아올 클라이언트들을 보관하는 리스트
    self.teacher_models = [] #받아온 모델 보관하는 리스트 
    self.dataloader = dataloader #보유 데이터셋을 pytorch에서 사용하는 dataloader형으로 
    self.testloader = testloader
    self.params = None #클라이언트가 가지고 있는 파라미터들 
    self.model = model_type() #학습을 위해 존재하는 구조 
    self.model_type = model_type #구조 클래스를 나타냄, student 가 사용
    self.batch_size = batch_size


  def get_teacher_models(self, inQ, outQ):
    # request teachers models
    while True:
      outQ.put({'type': 'read', 'from': self.clientID, 'what': self.teachers})
      while True:
        time.sleep(0.1)
        if not inQ.empty():
          break
      msg = inQ.get()
      if msg['status'] == 'success':
        break

    # make all teacher models 
    for data in msg['data']:
      TM = data['model_type']()
      TM.load_state_dict(data['params'])
      self.teacher_models.append(TM)


  def local_train(self, inQ, outQ, n_epochs=1):
    print(str(self.clientID) + " training with " + str(len(self.dataloader)) + " data")
    
    self.model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
      train_loss = 0
      total = 0
      correct = 0

      for batch_idx, data in enumerate(self.dataloader):
        image, label = data
 
        # Grad initialization
        optimizer.zero_grad()
        # Forward propagation
        output = self.model(image)
        # Calculate loss
        loss = criterion(output, label)
        # Backprop
        loss.backward()
        # Weight update
        optimizer.step()

        train_loss += loss.item()

        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

      print("Step: {}/{} | Acc:{:.3f}%".format(batch_idx + 1, len(self.dataloader),
                                                                                  100. * correct / total))

    outQ.put({'type': 'write', 'from': self.clientID, 'data':{'model_type': self.model_type, 'params': self.model.state_dict()}})
    while True:
      time.sleep(0.1)
      if not inQ.empty():
        break
    msg = inQ.get()


  
  def KD_trainNtest(self, inQ, outQ, n_epochs=1):
    student = self.model
    student.train()  # tells student to do training

    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)


    for epoch in range(n_epochs):
      # for log
      nProcessed = 0
      nTrain = len(self.dataloader.dataset)

      for batch_idx, data in enumerate(self.dataloader):
          image, label = data

          # randomly select teacher for each batch
          teacher = self.teacher_models[np.random.randint(0,len(self.teacher_models))]
          # sets gradient to 0
          optimizer.zero_grad()

          # forward, backward, and opt
          outputs, teacher_outputs = student(image), teacher(image)
          # mean = (F.one_hot(label, num_classes=10)+outputs)/2
          # dist = torch.norm(mean-teacher_outputs)
          dist = torch.norm(F.one_hot(label, num_classes=10)-teacher_outputs)
          alpha = (math.sqrt(2) - dist)/math.sqrt(2)
          # alpha = 0.9
          temperature = ()
          loss = KD.criterion_KD(outputs, label, teacher_outputs, alpha=alpha, temperature=temperature)
          loss.backward()
          optimizer.step()

          # for log
          nProcessed += len(image)
          pred = outputs.data.max(1)[1]  # get the index of the max log-probability
          incorrect = pred.ne(label.data).cpu().sum()  # ne: not equal
          err = 100. * incorrect / len(image)
          partialEpoch = epoch + batch_idx / len(self.dataloader)

          # print at STDOUT
          # print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
          #     partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(self.dataloader), loss.item(), err
          # ))
<<<<<<< HEAD
    self.params = self.model.state_dict()    
=======
    outQ.put({'type': 'write', 'from': self.clientID, 'data':{'model_type': self.model_type, 'params': self.model.state_dict()}})
    while True:
      time.sleep(0.1)
      if not inQ.empty():
        break
    msg = inQ.get()

    return test.test(self)
    
>>>>>>> a36e4f04a79c4149c75dcaf589f2701e4617b298
