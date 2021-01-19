import torch
import torch.nn as nn
import numpy as np

import parameters as pm

class Client():
    def __init__(self, clientID, model, server, dataloader, n_chunks, byzantine=False):
        self.clientID = clientID
        self.module = model()
        self.server = server
        self.params= pm.Parameters(server.download_params())
        self.dataloader = dataloader
        self.n_chunks = n_chunks
        self.byzantine = byzantine

    def download_params(self):
        self.params.paramTensor = (self.server.download_params() + self.params.paramTensor)/2
        print(self.clientID + " downloaded_parameters from the server")
    

    def train(self, n_epochs):
        print(self.clientID + " training with " + str(len(self.dataloader)) + " data")
        self.module.load_state_dict(self.params.getAsDict(self.module.state_dict()))
        for epoch in range(n_epochs):
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

            self.module.train()
            train_loss = 0
            total = 0
            correct = 0

            for batch_idx, data in enumerate(self.dataloader):
                image, label = data
                if self.byzantine:
                    label = 9-label
                # Grad initialization
                optimizer.zero_grad()
                # Forward propagation
                output = self.module(image)
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
        
        self.params.setByDict(self.module.state_dict())


    def bid(self):
        grad = (self.params.paramTensor - self.server.download_params())
        chunks = []
        for i in range(self.n_chunks):
            chunks.append(grad[int(i/self.n_chunks*len(grad)) : int((i+1)/self.n_chunks*len(grad))])
        
        indicies = np.random.permutation(np.arange(10))[:2]
        for index in indicies:
            self.server.upload({index: (torch.norm(chunks[index]), self)})
    

    def push(self, chunk_num):
        chunk = self.params.paramTensor[int(chunk_num/self.n_chunks*len(self.params.paramTensor)) : int((chunk_num+1)/self.n_chunks*len(self.params.paramTensor))]
        self.server.push(chunk_num, chunk)