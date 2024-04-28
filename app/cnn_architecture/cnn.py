import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataclasses import dataclass
from typing import Any


@dataclass
class CNNConfig:

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

#TODO same_padding confige gore guncellenecek. 
#TODO init kismindeki degerler tomldan cekilecek.

class CNN(nn.Module):


	def __init__(self , config):
		super(CNN,self).__init__()


		self.relu = nn.ReLU()

		self.cnn1 = nn.Conv2d(in_channels=config.image_number_of_channel , out_channels=config.layer1_number_of_channels, kernel_size=config.layer1_filter_kernel_size, stride=config.layer1_stride , padding=self.same_padding(32,3,1))
		self.batchnorm1 = nn.BatchNorm2d(config.layer1_number_of_channels)
		self.cnn2 = nn.Conv2d(in_channels=config.layer1_number_of_channels , out_channels=config.layer2_number_of_channels, kernel_size=config.layer2_filter_kernel_size, stride=config.layer2_stride  , padding=self.same_padding(32,3,1))
		self.batchnorm2 = nn.BatchNorm2d(config.layer2_number_of_channels)
		self.maxpool1 = nn.MaxPool2d(kernel_size=config.first_max_pool_kernel_size)
		self.cnn3 = nn.Conv2d(in_channels=config.layer2_number_of_channels , out_channels=config.layer3_number_of_channels, kernel_size=config.layer3_filter_kernel_size, stride=config.layer3_stride, padding=config.layer3_padding)
		self.batchnorm3 = nn.BatchNorm2d(config.layer3_number_of_channels)
		self.maxpool2 = nn.MaxPool2d(kernel_size=config.second_max_pool_kernel_size)
		self.fc1 = nn.Linear(in_features = config.fc1_in_features , out_features = config.fc1_out_features)
		self.dropout = nn.Dropout(p=config.dropout_p)
		self.fc2 = nn.Linear(in_features = config.fc1_out_features , out_features = config.fc2_out_features)


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

	config_dict = {
		"image_number_of_channel" : 3,
    
		"layer1_filter_kernel_size" : 3,
		"layer1_stride" : 1,
		"layer1_padding" : "same",
		"layer1_number_of_channels" : 32,

		"layer2_filter_kernel_size" : 3,
		"layer2_stride" : 1,
		"layer2_padding" : "same",
		"layer2_number_of_channels" : 64,
		"first_max_pool_kernel_size" : 2,

		"layer3_filter_kernel_size" : 3,
		"layer3_stride" : 1,
		"layer3_padding" : 1,
		"layer3_number_of_channels" : 128,
		"second_max_pool_kernel_size" : 2,

		"fc1_in_features" : 8192,
		"fc1_out_features" : 4096,

		"dropout_p" : 0.35,

		"fc2_out_features" : 10

	}

	mean_rgb = (0.49139968,0.48215827,0.44653124)
	# mean_rgb = (0.5, 0.5, 0.5)

	stddev_rgb = (0.24703233, 0.24348505, 0.26158768)
	# stddev_rgb = (0.5, 0.5, 0.5)



	# convert data to a normalized torch.FloatTensor
	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(), # randomly flip and rotate
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean_rgb, stddev_rgb)
		])

	# choose the training and test datasets
	# train_data = datasets.CIFAR10('data', train=True,
	#                               download=False, transform=transform)
	# test_data = datasets.CIFAR10('data', train=False,
	#                              download=False, transform=transform)










	# Transfrom the images to Tensors
	# transforms_original = transforms.Compose([transforms.ToTensor(),
											#   transforms.Normalize(mean_rgb, stddev_rgb)])
	# 
	# transforms_photo = transforms.Compose([transforms.Resize((32,32)),
										#   transforms.ToTensor(),
										#   transforms.Normalize(mean_rgb, stddev_rgb)])

	train_dataset = datasets.CIFAR10(root="./data",
								train = True,
								transform = transform,
								download = False)

	test_dataset = datasets.CIFAR10(root="./data",
								train = False,
								transform = transform)

	# unnormalized_image = unnormalize(train_dataset[200][0], mean_rgb, stddev_rgb)
	# 
	# plt.imshow(np.transpose(unnormalized_image.numpy(), (1, 2, 0)))
	# plt.show()

	batch_size = 200

	train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
											batch_size = batch_size,
											shuffle = True)

	test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
											batch_size = batch_size,
											shuffle = False)
	
	config = CNNConfig(**config_dict)
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




