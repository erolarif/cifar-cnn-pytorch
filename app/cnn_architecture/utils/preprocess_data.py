from dataclasses import dataclass
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from typing import Any

#TODO metodlara acıklama eklenecek
#TODO input outputlara type verilecek
@dataclass
class PreProcess:
    dataset_path : str
    random_rotation_value : int
    batch_size : int
    download_dataset : bool
    logger:Any=None


    def __call__(self):

        cifar_trainset = datasets.CIFAR10(root=self.dataset_path, train=True, download=self.download_dataset, transform=transforms.ToTensor())
        result = self.calc_mean_and_std(cifar_trainset)
        
        transform = self.augmenting_data(result)

        train_loader , test_loader , train_dataset , test_dataset= self.create_loader(transform, result)

        return train_loader , test_loader , train_dataset , test_dataset

        
    def calc_mean_and_std(self,cifar_trainset):
        

        imgs = [item[0] for item in cifar_trainset] # item[0] and item[1] are image and its label
        imgs = torch.stack(imgs, dim=0).numpy()

        # calculate mean over each channel (r,g,b)
        mean_r = imgs[:,0,:,:].mean()
        mean_g = imgs[:,1,:,:].mean()
        mean_b = imgs[:,2,:,:].mean()
        self.logger.info(f"mean_r: {mean_r},mean_g: {mean_r},mean_b: {mean_b}")

        # calculate std over each channel (r,g,b)
        std_r = imgs[:,0,:,:].std()
        std_g = imgs[:,1,:,:].std()
        std_b = imgs[:,2,:,:].std()
        self.logger.info(f"std_r: {std_r},std_g: {std_r},std_b: {std_b}")

        result = ((mean_r,mean_g,mean_b) , (std_r,std_g,std_b))

        return result
    
    def augmenting_data(self,mean_stddev):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(self.random_rotation_value),
        transforms.ToTensor(),
        transforms.Normalize(mean_stddev[0] , mean_stddev[1])
        ])

        return transform
    
    def create_loader(self,transform, mean_stddev):
        train_dataset = datasets.CIFAR10(root=self.dataset_path,
                                    train = True,
                                    transform = transform,
                                    download = False)

        test_dataset = datasets.CIFAR10(root=self.dataset_path,
                                    train = False,
                                    transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean_stddev[0] , mean_stddev[1])
                                                ]))

        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True)

        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = False)
        
        result = train_loader , test_loader , train_dataset , test_dataset
        return result