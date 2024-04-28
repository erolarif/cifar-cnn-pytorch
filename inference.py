from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset


from typing import List


@dataclass
class ModelInference():
    model_path : str
    dataset_path : str
    mean_rgb : List
    stddev_rgb : List
    

    def __post_init__(self):
        self.model = torch.load(self.model_path)

        self.transforms_photo = transforms.Compose([transforms.Resize((32,32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(tuple(self.mean_rgb), tuple(self.stddev_rgb))])
        
        self.test_dataset = datasets.CIFAR10(root=self.dataset_path,
                                    train = False,
                                    transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(self.mean_rgb , self.stddev_rgb)
                                                ]))
        
        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset,
                                                batch_size = 1,
                                                shuffle = False)

    def predict(self,image):


        image = image.unsqueeze(0).cuda()
        output = self.model(image)

        _,predicted = torch.max(output , 1)

        return predicted.item()
    
    def unnormalize(self,image):
        """
        Unnormalize a normalized image.
        
        Args:
        image (Tensor): Normalized image tensor of shape (C, H, W) to be unnormalized.
        mean (tuple): Mean values for each channel to be added.
        std (tuple): Standard deviation values for each channel to be multiplied.
        
        Returns:
        Tensor: Unnormalized image tensor.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch Tensor.")

        if len(image.shape) != 3:
            raise ValueError("Input image must be of shape (C, H, W).")

        if len(self.mean_rgb) != 3 or len(self.stddev_rgb) != 3:
            raise ValueError("Mean and std must be tuples of length 3.")

        # Clone the tensor to avoid modifying the original tensor
        image = image.clone()

        for i in range(3):  # Assuming RGB image
            image[i] = image[i] * self.stddev_rgb[i] + self.mean_rgb[i]

        return image



if __name__ == "__main__":
    dict = {
        "model_path" : "/home/ak/workplace/cifar-cnn-pytorch/model_best.pt",
        "dataset_path" : "./app/data",
        "mean_rgb" : [0.49139968,0.48215827,0.44653124],
        "stddev_rgb" : [0.24703233, 0.24348505, 0.26158768]
    }

    classes = [
        "airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"
    ]
    inference = ModelInference(**dict)
    imshow_img = inference.unnormalize(inference.test_dataset[200][0])
    plt.imshow(np.transpose(imshow_img.numpy(), (1, 2, 0)))
    plt.show()
    random_image = inference.test_dataset[200][0]

    prediction = inference.predict(random_image)

    print("Label : " , classes[inference.test_dataset[200][1]])

    print("prediction = " , classes[prediction])
