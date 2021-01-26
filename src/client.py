import torch
import torch.nn as nn
import numpy as np

import parameters as pm

class Client():
    def __init__(self, clientID, teachers, dataloader, model_type):
        #(self, clientID, model, dataloader, n_chunks, byzantine=False):  
        '''
        self.clientID = clientID
        self.module = model()
        self.params= pm.Parameters(server.download_params())
        self.dataloader = dataloader
        self.n_chunks = n_chunks
        self.byzantine = byzantine
        '''

        self.clientID = clientID 
        self.teachers = teachers # 모델 받아올 클라이언트들을 보관하는 리스트
        self.teacher_models = [] #받아온 모델 보관하는 리스트 
        self.dataloader = dataloader #보유 데이터셋을 pytorch에서 사용하는 dataloader형으로 
        self.params = None #클라이언트가 가지고 있는 파라미터들 
        self.model = model_type() #학습을 위해 존재하는 구조 
        self.model_type = model_type #구조 클래스를 나타냄, student 가 사용
        


        def local_train(self):
            #로칼에서 1회 트레이닝 


            self.params = self.model.state_dict()



        def KD_train(self) : 
            #클라이언트 내부에서 지식증류 




        def get_teacher_models(self):
            #teachers의 parameter와 model_type을 받아서 model을  만든 후 teachermodels에 저장하는 역할
            for teacher in self.teachers :
                TM = teacher.model_type()
                TM.load_state_dict(teacher.params)
                self.teacher_models.append(TM)

