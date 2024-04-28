from dataclasses import dataclass
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Any


#TODO logger gelecek
#TODO metodlara acıklama eklenecek
#TODO input outputlara type verilecek
@dataclass
class PostProcess:
    number_of_epoch : int
    learning_rate : float
    momentum : float
    parameters:Any=None
    correct = 0
    iterations = 0
    iter_loss = 0
    testing_loss = 0.0
    logger:Any=None

    def __post_init__(self,):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=self.momentum)
    
    @staticmethod
    def check_cuda_availibility():
        cuda_availibility = torch.cuda.is_available()
        return cuda_availibility
    
    def optimize_model(self,outputs,labels):
        loss = self.loss_fn(outputs,labels)
        self.iter_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()      #update the weights
    
    def calc_accuracy_for_each_iteration(self,outputs,labels):
        _, predicted = torch.max(outputs , 1)
        self.correct += (predicted == labels).sum()
        self.iterations += 1

    def save_model(self,model, test_loss,epoch):
        if epoch <= 0:
            return
        if test_loss[-1] > test_loss[-2]:
            return
        torch.save(model, "model_best.pt")
        self.logger.info("Model saved due to decrease test loss from the previous epoch.")

    def __call__(self,model,train_loader,test_loader,train_dataset,test_dataset):
        if PostProcess.check_cuda_availibility():
            model = model.cuda()

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        for epoch in range(self.number_of_epoch):
            self.correct = 0
            self.iterations = 0
            self.iter_loss = 0

            model.train()

            for i , (inputs,labels) in enumerate(train_loader):
                

                if PostProcess.check_cuda_availibility():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)

                self.optimize_model(outputs , labels)

                self.calc_accuracy_for_each_iteration(outputs , labels)

            train_loss.append(self.iter_loss/self.iterations)
            train_accuracy.append((100*self.correct/len(train_dataset)))

            self.testing_loss = 0.0
            self.correct = 0
            self.iterations = 0

            model.eval()

            for i , (inputs,labels) in enumerate(test_loader):

                if PostProcess.check_cuda_availibility():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                loss = self.loss_fn(outputs , labels)
                self.testing_loss += loss.item()

                self.calc_accuracy_for_each_iteration(outputs , labels)

            test_loss.append(self.testing_loss/self.iterations)
            test_accuracy.append((100*self.correct/len(test_dataset)))

            self.logger.info('Epoch{}/{}, Training Loss: {:.3f}, Training Accuracy : {:.3f}, Testing Loss: {:.3f}, Testing Accuracy : {:.3f}'.format(epoch+1 , self.number_of_epoch, train_loss[-1], train_accuracy[-1], test_loss[-1] , test_accuracy[-1]))
            
            self.save_model(model,test_loss,epoch)

