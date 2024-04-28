from app.config import Configs
from app.logger import Logger
from app.cnn import CNN
from app.cnn import CNNConfig
from app.preprocess_data import PreProcess
from app.post_process import PostProcess

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


def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ [NAME OF PROJECT] CONFIGURATIONS ############")
    logger.debug(configs)
    print(args)

    preprocess = PreProcess(**configs["preprocess"])
    train_loader , test_loader , train_dataset , test_dataset = preprocess()
    config = CNNConfig(**configs["cnn-architecture"])
    model = CNN(config)
    
    CUDA = torch.cuda.is_available()

    if CUDA:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.2)

    num_epochs = 25

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        correct = 0
        iterations = 0
        iter_loss = 0

        model.train()

        for i , (inputs,labels) in enumerate(train_loader):
            CUDA = torch.cuda.is_available()

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            iter_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      #update the weights

            _, predicted = torch.max(outputs , 1)
            correct += (predicted == labels).sum()
            iterations += 1

        train_loss.append(iter_loss/iterations)
        train_accuracy.append((100*correct/len(train_dataset)))

        testing_loss = 0.0
        correct = 0
        iterations = 0

        model.eval()

        for i , (inputs,labels) in enumerate(test_loader):
            CUDA = torch.cuda.is_available()

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs , labels)
            testing_loss += loss.item()

            _, predicted = torch.max(outputs , 1)
            correct += (predicted == labels).sum()
            iterations += 1

        test_loss.append(testing_loss/iterations)
        test_accuracy.append((100*correct/len(test_dataset)))

        print('Epoch{}/{}, Training Loss: {:.3f}, Training Accuracy : {:.3f}, Testing Loss: {:.3f}, Testing Accuracy : {:.3f}'.format(epoch+1 , num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1] , test_accuracy[-1]))



  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
