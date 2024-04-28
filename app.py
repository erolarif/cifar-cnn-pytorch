from app.config import Configs
from app.logger import Logger
from app.cnn_architecture.cnn import CNN
from app.cnn_architecture.cnn_2 import CNN2

from app.cnn_architecture.cnn import CNNConfig
from app.cnn_architecture.utils.preprocess_data import PreProcess
from app.cnn_architecture.utils.post_process import PostProcess

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
    #TODO Model Saver yazılması
    #TODO Graph creater
    preprocess = PreProcess(**configs["preprocess"])
    train_loader , test_loader , train_dataset , test_dataset = preprocess()
    config = CNNConfig(**configs["cnn-architecture"])
    # model = CNN(config)
    model = CNN2()

    post_process = PostProcess(**configs["postprocess"] , parameters = model.parameters())

    post_process(model,train_loader,test_loader,train_dataset,test_dataset)
    
    
  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
