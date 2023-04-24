import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2

class ConvNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ConvNet, self).__init__()

        #Input shape = (256, 3, 500, 500)
        # (w-f+2P)/s + 1
        # w = width = 500
        # f = kernel_size = 3
        # P = padding = 1
        # S = stride = 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)        
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.conv2= nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2=nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()

        self.fc=nn.Linear(in_features=32*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = output.view(-1, 32*75*75)
        output = self.fc(output)
        return output

def infer(train_path, num_classes, checkpoint_name):    
    root=pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])


    checkpoint = torch.load(checkpoint_name)
    model=ConvNet(num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.eval()

    transformer = transforms.Compose([
        transforms.Resize((150,150)),    
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    def prediction(img_path, transformer):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transformer(image).float()

        image_tensor = image_tensor.unsqueeze_(0)

        if(torch.cuda.is_available()):
            image_tensor.cuda()

        input = Variable(image_tensor)
        output = model(input)

        index = output.data.numpy().argmax()
        pred = classes[index]

        return pred


    image_to_predict = ""
    while(image_to_predict != "quit"):
        image_to_predict = input("Enter image path:")
        if(image_to_predict != "quit"):
            print(str(prediction(image_to_predict, transformer)))

def main():
    train_path_all = "..\\DataCollection\\DataSetAllMons"
    train_path_generational = "..\\DataCollection\\DataSetGenerational"
    train_path_generation1 = "..\\DataCollection\\DataSetGeneration1"
    train_path_types = "..\\DataCollection\DataSetTypes"

    print("1. All Mons")
    print("2. Generational")
    print("3. Generation One")
    print("4. Types")
    choice = input("Enter your model to infer with:")    

    if(int(choice) == 1):
        infer(train_path_all, 1010, 'all_checkpoint.model')
    if(int(choice) == 2):
        infer(train_path_generational, 9, 'generational_checkpoint.model')
    if(int(choice) == 3):
        infer(train_path_generation1, 150, 'generation1_checkpoint.model')
    if(int(choice) == 4):
        infer(train_path_types, 18, 'types_checkpoint.model')

main()