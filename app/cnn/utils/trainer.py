import os
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


@dataclass
class Trainer:
    save_model_path:str
    number_of_epoch : int
    learning_rate : float
    momentum : float
    parameters:Any=None
    correct:int = 0
    iterations:int = 0
    iter_loss:int = 0
    testing_loss:float = 0.0
    logger:Any=None

    def __post_init__(self,) -> None:
        """
        Initialize and decide loss_func and optimization algorithm
        """
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=self.momentum)
    
    @staticmethod
    def check_cuda_availibility():
        """
        Control the cuda availibility

        Returns
        cuda_availibility: bool
        """
        cuda_availibility = torch.cuda.is_available()
        return cuda_availibility
    
    def optimize_model(self,outputs: Any,labels:Any) -> None:
        """
        Backpropagation and update the weights
        
        Args
        outputs: result of forward pass
        labels: label of image 
        """
        loss = self.loss_fn(outputs,labels)
        self.iter_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def calc_accuracy_for_each_iteration(self,outputs:Any,labels:Any)-> None:
        """
        Find predicted_label and update correct predicted count
                
        Args
        outputs: result of forward pass
        labels: label of image 
        """
        _, predicted = torch.max(outputs , 1)
        self.correct += (predicted == labels).sum()
        self.iterations += 1
    
    def draw_loss_graph(self,train_loss:Any,test_loss:Any)->None:
        """
        draw loss graph end of the train

        Args
        train_loss : loss list of train results
        test_loss : test list of test results
        """
        fig = plt.figure(figsize = (10,10))
        plt.plot(train_loss , label = 'Training Loss')
        plt.plot(test_loss , label = 'Testing Loss')
        plt.legend()
        plt.savefig('./logs/loss_plot.png')  # Grafiği 'loss_plot.png' olarak kaydediyoruz
        plt.show()

    def save_model(self,model:Any, test_loss:list ,epoch:int) -> None:
        """
        Save model if loss decreae according to tthe previous epoch
        
        Args
        test_loss : test list of test results
        epoch : number of saved epoch 
        """
        if epoch <= 0:
            return
        if test_loss[-1] > test_loss[-2]:
            return
        if not os.path.exists(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        torch.save(model, self.save_model_path)
        self.logger.info("Model saved due to decrease test loss from the previous epoch.")

    def __call__(self,model:Any,train_loader:list,test_loader:list,train_dataset:list,test_dataset:list)-> None:
        """
        Train model with __call__ method.
        """
        if Trainer.check_cuda_availibility():
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
                

                if Trainer.check_cuda_availibility():
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

                if Trainer.check_cuda_availibility():
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
        self.draw_loss_graph(train_loss,test_loss)

