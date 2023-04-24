import os
import csv
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

class ConvNet(nn.Module):
    def __init__(self, num_classes=1010):
        super(ConvNet, self).__init__()

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


def train(num_epochs, train_path, test_path, num_classes, model_name, generation_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
        

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=256, shuffle=True
    )

    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=256, shuffle=True
    )

    root=pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

    print(classes)

    model = ConvNet(num_classes=num_classes).to(device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()    

    train_count = len(glob.glob(train_path+'\\**\\*'))
    test_count = len(glob.glob(test_path+'\\**\\*'))

    print(train_count)
    print(test_count)

    best_accuracy = 0.0
    for epoch in range(num_epochs):    
        train_accuracy = 0.0
        test_accuracy = 0.0
        train_loss = 0.0
        for i, (images,labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())
            
            optimizer.zero_grad()

            outputs = model(images)
            loss=loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data*images.size(0)
            _,prediction=torch.max(outputs.data,1)

            train_accuracy+=int(torch.sum(prediction==labels.data))
        
        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count

        model.eval()
        for i, (images,labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())

            outputs = model(images)
            _,prediction = torch.max(outputs.data,1)
            test_accuracy+=int(torch.sum(prediction==labels.data))

        test_accuracy=test_accuracy/test_count

        print("Epoch " + str(epoch+1) + " - Train Loss: " + str(train_loss) + " Train Accuracy: " + str(train_accuracy) + " Test Accuracy: " + str(test_accuracy))
        
        
        fieldnames = ["Epoch", "Loss", "Train Accuracy", "Test Accuracy"]                
        data = {"Epoch": epoch+1, "Loss": str(float(train_loss)), "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy}
        
        with open("results-" + generation_type + ".csv", mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(data)

        if(test_accuracy > best_accuracy):
            torch.save(model.state_dict(), model_name + '.model')
            best_accuracy = test_accuracy

def main():
    train_path_all = "..\\DataCollection\\DataSetAllMons"
    train_path_generational = "..\\DataCollection\\DataSetGenerational"
    train_path_generation1 = "..\\DataCollection\\DataSetGeneration1"
    train_path_types = "..\\DataCollection\DataSetTypes"

    test_path_all = "..\\DataCollection\\DataSetAllMonsTesting"
    test_path_generational = "..\\DataCollection\\DataSetGenerationalTesting"
    test_path_generation1 = "..\\DataCollection\\DataSetGeneration1Testing"
    test_path_types = "..\\DataCollection\DataSetTypesTesting"

    num_classes_all = 1010
    num_classes_generational = 9
    num_classes_generation1 = 150

    print("1. All Mons")
    print("2. Generational")
    print("3. Generation One")
    print("4. Types")
    choice = input("Enter your model to train:")    

    if(int(choice) == 1):
        train(50, train_path_all, test_path_all, 1010, 'all_checkpoint', 'all')
    if(int(choice) == 2):
        train(50, train_path_generational, test_path_generational, 9, 'generational_checkpoint', 'generational')
    if(int(choice) == 3):
        train(50, train_path_generation1, test_path_generation1, 150, 'generation1_checkpoint', 'generation1')
    if(int(choice) == 4):
        train(50, train_path_types, test_path_types, 18, 'types_checkpoint', 'types')

main()