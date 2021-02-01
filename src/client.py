import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def criterion_KD(
    outputs,
    labels,
    teacher_outputs,
    alpha: float = 0.1,
    temperature: float = 3.
):
    loss_KD = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1)
    ) * (alpha * temperature * temperature) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)
    return loss_KD

class Client:
  def __init__(self, clientID, dataloader, model_type):
    self.clientID = clientID 
    self.teachers = None # 모델 받아올 클라이언트들을 보관하는 리스트
    self.teacher_models = [] #받아온 모델 보관하는 리스트 
    self.dataloader = dataloader #보유 데이터셋을 pytorch에서 사용하는 dataloader형으로 
    self.params = None #클라이언트가 가지고 있는 파라미터들 
    self.model = model_type() #학습을 위해 존재하는 구조 
    self.model_type = model_type #구조 클래스를 나타냄, student 가 사용


  def get_teacher_models(self):
    for teacher in self.teachers :
      TM = teacher.model_type()
      TM.load_state_dict(teacher.params)
      self.teacher_models.append(TM)


  def local_train(self, n_epochs=1):
    print(self.clientID + " training with " + str(len(self.dataloader)) + " data")
    for epoch in range(n_epochs):
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)


      self.model.train()
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
    self.params = self.model.state_dict()

  
  def KD_train(self, n_epochs=3):
    self.model.load_state_dict(self.params)
    student = self.model
    student.train()  # tells student to do training

    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)


    for epoch in range(n_epochs):

      # for log
      nProcessed = 0
      nTrain = len(self.dataloader.dataset)

      for batch_idx, (inputs, targets) in enumerate(self.dataloader):
          # randomly select teacher for each batch

          teacher = self.teacher_models[np.random.randint(0,len(self.teacher_models))]
          # sets gradient to 0
          optimizer.zero_grad()

          # forward, backward, and opt
          outputs, teacher_outputs = student(inputs), teacher(inputs)
          loss = criterion_KD(outputs, targets, teacher_outputs)
          loss.backward()
          optimizer.step()

          # for log
          nProcessed += len(inputs)
          pred = outputs.data.max(1)[1]  # get the index of the max log-probability
          incorrect = pred.ne(targets.data).cpu().sum()  # ne: not equal
          err = 100. * incorrect / len(inputs)
          partialEpoch = epoch + batch_idx / len(self.dataloader)

          # print at STDOUT
          # print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
          #     partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(self.dataloader), loss.item(), err
          # ))
    self.params = self.model.state_dict()
    
