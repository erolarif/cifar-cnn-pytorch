import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataclasses import dataclass
from typing import Any


@dataclass
class CNNUniqueConfig:

	image_number_of_channel: int

	layer1_filter_kernel_size: int
	layer1_stride: int
	layer1_padding: int
	layer1_number_of_channels: int

	layer2_filter_kernel_size: int
	layer2_stride: int
	layer2_padding: int
	layer2_number_of_channels: int
	first_max_pool_kernel_size: int

	layer3_filter_kernel_size: int
	layer3_stride: int
	layer3_padding: int
	layer3_number_of_channels: int
	second_max_pool_kernel_size: int
	fc1_in_features: int
	fc1_out_features: int
	dropout_p: float
	fc2_out_features: int

	softmax_dim : int

#TODO same_padding confige gore guncellenecek. 
#TODO init kismindeki degerler tomldan cekilecek.

class UniqueCNN(nn.Module):


	def __init__(self , config):
		super(UniqueCNN,self).__init__()


		self.relu = nn.ReLU()

		self.cnn1 = nn.Conv2d(in_channels=config.image_number_of_channel , out_channels=config.layer1_number_of_channels, kernel_size=config.layer1_filter_kernel_size, stride=config.layer1_stride ,
						 padding=self.same_padding(32,config.layer1_filter_kernel_size,config.layer1_stride))
		self.batchnorm1 = nn.BatchNorm2d(config.layer1_number_of_channels)
		self.cnn2 = nn.Conv2d(in_channels=config.layer1_number_of_channels , out_channels=config.layer2_number_of_channels, kernel_size=config.layer2_filter_kernel_size, stride=config.layer2_stride  ,
						 padding=self.same_padding(32,config.layer2_filter_kernel_size,config.layer2_stride))
		self.batchnorm2 = nn.BatchNorm2d(config.layer2_number_of_channels)
		self.maxpool1 = nn.MaxPool2d(kernel_size=config.first_max_pool_kernel_size)
		self.cnn3 = nn.Conv2d(in_channels=config.layer2_number_of_channels , out_channels=config.layer3_number_of_channels, kernel_size=config.layer3_filter_kernel_size, stride=config.layer3_stride, padding=config.layer3_padding)
		self.batchnorm3 = nn.BatchNorm2d(config.layer3_number_of_channels)
		self.maxpool2 = nn.MaxPool2d(kernel_size=config.second_max_pool_kernel_size)
		self.fc1 = nn.Linear(in_features = config.fc1_in_features , out_features = config.fc1_out_features)
		self.dropout = nn.Dropout(p=config.dropout_p)
		self.fc2 = nn.Linear(in_features = config.fc1_out_features , out_features = config.fc2_out_features)
		self.softmax = nn.Softmax(dim=config.softmax_dim)


		# self.relu = nn.ReLU()

		# self.cnn1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
		# self.batchnorm1 = nn.BatchNorm2d(32)

		# self.cnn2 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
		# self.batchnorm2 = nn.BatchNorm2d(64)
		# self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		# self.cnn3 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=3, stride=1, padding=1)
		# self.batchnorm3 = nn.BatchNorm2d(128)
		# self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		# self.fc1 = nn.Linear(in_features = 8192 , out_features = 4096)
		# self.dropout = nn.Dropout(p=0.35)
		# self.fc2 = nn.Linear(in_features = 4096 , out_features = 10)


	def forward(self,x : Any) -> Any:
		"""
		Forward Propogation of the model

		Args 

		x : Input of the cnn
		"""
		x = self.cnn1(x)
		x = self.batchnorm1(x)
		x = self.relu(x)
		x = self.cnn2(x)
		x = self.batchnorm2(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.cnn3(x)
		x = self.batchnorm3(x)
		x = self.relu(x)

		x = self.maxpool2(x)

		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
    
	def same_padding(self,input_size, kernel_size, stride):
		"""
		This method calculates the padding value og the layer to do same padding
		"""
		
		padding = int((((stride -1) * input_size) - stride + kernel_size) / 2)
		# padding = ((input_size) * (stride - 1) + kernel_size - input_size)/2
		return padding


if __name__ == "__main__":

	pass
