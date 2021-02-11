# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:54:03 2021

@author: aksha
"""
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from torch import nn
import cv2

img = cv2.imread('frame%d.jpg',0)

cv2.imwrite('frame%d.jpg',img)



im = Image.open("frame%d.jpg")


im.show()

resized_im = im.resize((round(28), round(28)))


resized_im.show()


resized_im.save('resized.jpg')

PATH = './MNIST_net.pth'

imsize = 28

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.convolutaional_neural_network_layers = nn.Sequential(
            
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28) 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), 
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2) 
        )

        self.linear_layers = nn.Sequential(

                nn.Linear(in_features=24*7*7, out_features=64),          
                nn.ReLU(),
                nn.Dropout(p=0.2), 
                nn.Linear(in_features=64, out_features=10)  
        )

    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
model_ft = Network()
model_ft.load_state_dict(torch.load(PATH))


print(np.argmax(model_ft(image_loader(data_transforms, 'frame%d.jpg')).detach().numpy()))