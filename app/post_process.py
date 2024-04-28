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

#TODO logger gelecek
#TODO metodlara acıklama eklenecek
#TODO input outputlara type verilecek
@dataclass
class PostProcess:
    number_of_epoch : int
    learning_rate : float
    momentum : float

    def __post_init__(self,parameters):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)




    def __call__():
        pass

