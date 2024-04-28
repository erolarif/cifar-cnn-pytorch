import torch.nn as nn

#TODO Config eklenecek
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.cnn2 = nn.Conv2d(in_channels=32 , out_channels=32, kernel_size=3, stride=1 , padding=self.same_padding(32,3,1))
        #RELU
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.cnn3 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=3, stride=1, padding=self.same_padding(16,3,1))
        #RELU
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.cnn4 = nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=3, stride=1 , padding=self.same_padding(16,3,1))
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.25)


        self.cnn5 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=3, stride=1 , padding=self.same_padding(8,3,1))
        #RELU
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.cnn6 = nn.Conv2d(in_channels=128 , out_channels=128, kernel_size=3, stride=1 , padding=self.same_padding(8,3,1))
        self.batchnorm6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.25)


        self.fc1 = nn.Linear(in_features = 2048 , out_features = 512)
        #RELU
        self.dropout4 = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
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
    
       
    def same_padding(self,input_size, kernel_size, stride):

       padding = int((((stride -1) * input_size) - stride + kernel_size) / 2)
       # padding = ((input_size) * (stride - 1) + kernel_size - input_size)/2
       return padding


