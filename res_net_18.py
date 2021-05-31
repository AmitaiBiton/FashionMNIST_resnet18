import glob
import time
from tkinter import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns
import gc
import os
import torchvision.models as models
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image


train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
 transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
 transforms.Compose([transforms.ToTensor()]))

print(len(train_set) , len(test_set))

train_set =list(train_set)[:]
test_set = list(test_set)[:]
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=100)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size= 100 )

print(len(train_loader) , len(test_loader))

model= models.resnet18( num_classes=10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
#model.fc = torch.nn.Linear(512 ,10)
print(model)

device = "cpu"

if (torch.cuda.is_available()):
    device = "cuda"


model.cuda()



loss_criterion = nn.CrossEntropyLoss()
#loss_criterion = nn.NLLLoss()

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    tmp =0
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the optimizer
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criterion(output, target)
        tmp = loss
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics so we see some progress
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    tmp =0
    list_w = []
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            tmp = test_loss
            #tmp = test_loss
            # Calculate the loss for this batch
            test_loss += loss_criterion(output, target).item()

            list_w.append(test_loss-tmp)
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # return average loss for the epoch
    return avg_loss , 100. * correct / len(test_loader.dataset) , list_w

torch.cuda.empty_cache()
# Train over 10 epochs (We restrict to 10 for time issues)

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)#70%
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#73%

Device = "cuda"
epochs = 10
print('Training on', device)
epoch_nums=[]
training_loss=[]
validation_loss=[]
accuracy = []

for epoch in range(1, epochs+1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss , accuracy_score  , W= test(model, device, test_loader)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    accuracy.append(accuracy_score)
plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

plt.plot(epoch_nums, accuracy)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()
W = np.array(W)
if len(W)==10:
    W = np.array(W)
    plt.plot(epoch_nums, W)
    plt.xlabel("No. of Iteration")
    plt.ylabel("loss - lossa after updata")
    plt.title("")
    plt.show()



