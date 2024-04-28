import torch.nn as nn 
from typing import Any

from dataclasses import dataclass

#TODO Config eklenecek
@dataclass
class CNNOptimizedConfig:
    image_number_of_channel: int

    layer1_filter_kernel_size: int
    layer1_stride: int
    layer1_padding: str
    layer1_number_of_channels: int

    layer2_filter_kernel_size: int
    layer2_stride: int
    layer2_padding: str
    layer2_number_of_channels: int

    first_max_pool_kernel_size: int
    dropout_1_p: float

    layer3_filter_kernel_size: int
    layer3_stride: int
    layer3_padding: str
    layer3_number_of_channels: int

    layer4_filter_kernel_size: int
    layer4_stride: int
    layer4_padding: str
    layer4_number_of_channels: int
    second_max_pool_kernel_size: int
    dropout_2_p: float

    layer5_filter_kernel_size: int
    layer5_stride: int
    layer5_padding: str
    layer5_number_of_channels: int

    layer6_filter_kernel_size: int
    layer6_stride: int
    layer6_padding: str
    layer6_number_of_channels: int
    third_max_pool_kernel_size: int
    dropout_3_p: float

    fc1_in_features: int
    fc1_out_features: int
    dropout_4_p: float
    fc2_out_features: int
    softmax_dim : int


class CNN2(nn.Module):

    

    def __init__(self , config):
        super(CNN2,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=config.image_number_of_channel , out_channels=config.layer1_number_of_channels, kernel_size=config.layer1_filter_kernel_size, stride=config.layer1_stride , padding=self.same_padding(32,3,1))
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(config.layer1_number_of_channels)

        self.cnn2 = nn.Conv2d(in_channels=config.layer1_number_of_channels , out_channels=config.layer2_number_of_channels, kernel_size=config.layer2_filter_kernel_size, stride=config.layer2_stride , padding=self.same_padding(32,3,1))
        #RELU
        self.batchnorm2 = nn.BatchNorm2d(config.layer2_number_of_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size = config.first_max_pool_kernel_size)
        self.dropout1 = nn.Dropout(p = config.dropout_1_p)
        
        self.cnn3 = nn.Conv2d(in_channels=config.layer2_number_of_channels , out_channels=config.layer3_number_of_channels, kernel_size=config.layer3_filter_kernel_size, stride=config.layer3_stride, padding=self.same_padding(16,3,1))
        #RELU
        self.batchnorm3 = nn.BatchNorm2d(config.layer3_number_of_channels)
        self.cnn4 = nn.Conv2d(in_channels=config.layer3_number_of_channels , out_channels=config.layer4_number_of_channels, kernel_size=config.layer4_filter_kernel_size, stride=config.layer4_stride , padding=self.same_padding(16,3,1))
        self.batchnorm4 = nn.BatchNorm2d(config.layer4_number_of_channels)
        self.maxpool2 = nn.MaxPool2d(kernel_size= config.second_max_pool_kernel_size)
        self.dropout2 = nn.Dropout(p = config.dropout_2_p)


        self.cnn5 = nn.Conv2d(in_channels=config.layer4_number_of_channels , out_channels=config.layer5_number_of_channels, kernel_size=config.layer5_filter_kernel_size, stride=config.layer5_stride , padding=self.same_padding(8,3,1))
        #RELU
        self.batchnorm5 = nn.BatchNorm2d(config.layer5_number_of_channels)
        self.cnn6 = nn.Conv2d(in_channels=config.layer5_number_of_channels , out_channels=config.layer6_number_of_channels, kernel_size=config.layer6_filter_kernel_size, stride=config.layer6_stride  , padding=self.same_padding(8,3,1))
        self.batchnorm6 = nn.BatchNorm2d(config.layer6_number_of_channels)
        self.maxpool3 = nn.MaxPool2d(kernel_size=config.third_max_pool_kernel_size)
        self.dropout3 = nn.Dropout(p=config.dropout_3_p)


        self.fc1 = nn.Linear(in_features = config.fc1_in_features , out_features = config.fc1_out_features)
        #RELU
        self.dropout4 = nn.Dropout(p=config.dropout_4_p)
        self.fc2 = nn.Linear(in_features=config.fc1_out_features, out_features=config.fc2_out_features)
        self.softmax = nn.Softmax(dim=config.softmax_dim)

        



        # self.cnn1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
        # self.relu = nn.ReLU()
        # self.batchnorm1 = nn.BatchNorm2d(32)

        # self.cnn2 = nn.Conv2d(in_channels=32 , out_channels=32, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
        # #RELU
        # self.batchnorm2 = nn.BatchNorm2d(32)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.dropout1 = nn.Dropout(p=0.25)
        
        # self.cnn3 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=3, stride=1, padding=self.same_padding(16,3,1))
        # #RELU
        # self.batchnorm3 = nn.BatchNorm2d(64)
        # self.cnn4 = nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=3, stride=1 , padding=self.same_padding(16,3,1))
        # self.batchnorm4 = nn.BatchNorm2d(64)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.dropout2 = nn.Dropout(p=0.25)


        # self.cnn5 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=3, stride=1 , padding=self.same_padding(8,3,1))
        # #RELU
        # self.batchnorm5 = nn.BatchNorm2d(128)
        # self.cnn6 = nn.Conv2d(in_channels=128 , out_channels=128, kernel_size=3, stride=1 , padding=self.same_padding(8,3,1))
        # self.batchnorm6 = nn.BatchNorm2d(128)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.dropout3 = nn.Dropout(p=0.25)


        # self.fc1 = nn.Linear(in_features = 2048 , out_features = 512)
        # #RELU
        # self.dropout4 = nn.Dropout(p=0.15)
        # self.fc2 = nn.Linear(in_features=512, out_features=10)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x : Any) -> Any:

        """
		Forward Propogation of the model

		Args 

		x : Input of the cnn
		"""

        x = self.cnn1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool1(x)
        # x = self.dropout1(x)


        x = self.cnn3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.cnn4(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool2(x)
        # x = self.dropout2(x)

        x = self.cnn5(x)
        x = self.relu(x)
        x = self.batchnorm5(x)
        x = self.cnn6(x)
        x = self.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool3(x)
        # x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        # x= self.softmax(x)

        return x
    
       
    def same_padding(self,input_size, kernel_size, stride) -> int:
       """
		This method calculates the padding value og the layer to do same padding

        Args

        input_size : width/height value of the image
        kernel_size : filter size
        stride : stride value
	   """

       padding = int((((stride -1) * input_size) - stride + kernel_size) / 2)
       # padding = ((input_size) * (stride - 1) + kernel_size - input_size)/2
       return padding


