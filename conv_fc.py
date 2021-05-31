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

train_set =list(train_set)[:]
test_set = list(test_set)[:]
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size= 100 )



#Making a method that return the name of class for the label number. ex. if the label is 5, we return Sandal.
def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

a = next(iter(train_loader))
a[0].size()
#print(a[0].size())
#print(len(train_set))#60000
image, label = next(iter(train_set))
#plt.imshow(image.squeeze(), cmap="gray")
#plt.show()
print(label)
demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(demo_loader))
images, labels = batch
print(type(images), type(labels))
print(images.shape, labels.shape)
grid = torchvision.utils.make_grid(images, nrow=10)

#plt.figure(figsize=(15, 20))
#plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, label in enumerate(labels):
    print(output_label(label), end=", ")


# Create a neural net class
class Net(nn.Module):
    # Defining the Constructor

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32 , 64, 3, 1)
        #self.conv3 = nn.Conv2d(64 , 128, 3, 1)
        #self.conv4 = nn.Conv2d(256,512 , 3, 1)
        # self.conv5 = nn.Conv2d(1024, 2048, 3, 1)

        #self.dropout1 = nn.Dropout(0.5)
        #self.dropout2 = nn.Dropout(0.5)
        #self.dropout3 = nn.Dropout(0.5)
        #self.dropout4 = nn.Dropout(0.5)
        # self.dropout5 = nn.Dropout(1)

        self.fc1 = nn.Linear( 196,32)  # 14*14*64
        self.fc2 = nn.Linear(  32,10)
        #self.fc3 = nn.Linear(32,10)
        #self.fc4 = nn.Linear(32,2 )
        # self.fc5 = nn.Linear(128, 2)
    def forward(self, x):

        #x = self.conv1(x) # 1+32-3 30  , 128-3+1=126
        #x = F.relu(x)

        #x = self.conv2(x) # 1+30-3=28 n  , 124
        #x = F.relu(x)

        #x = self.conv3(x)  # 1+28-3=26 n , 122
        #x = F.relu(x)

        #x = self.conv4(x)  # 1+28-3=24 , 1+122-3=120
        #x = F.relu(x)
        #x = self.conv5(x)  # 1+28-3=22
        #x = F.relu(x)

        x = F.max_pool2d(x, 2)#22:2-2+1=11 , 120/2 =60

        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        #x = self.dropout2(x)
        x = self.fc2(x)

        #x = self.dropout3(x)
        #x = self.fc3(x)

        #x = self.dropout4(x)
        #x = self.fc4(x)

        #x = self.dropout5(x)
        #x = self.fc5(x)

        output = F.log_softmax(x, dim=1)

        return output

device = "cpu"

if (torch.cuda.is_available()):
    device = "cuda"

classes = 10
model = Net(num_classes=classes).to(device)
print(torch.cuda.is_available())
print(model)


loss_criteria = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the optimizer
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criteria(output, target)
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()
        # Print metrics so we see some progress
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    #print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    tmp = 0
    list_w = []
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            tmp = test_loss
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
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
Network = Net()
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
    test_loss , accuracy_score , W = test(model, device, test_loader)
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

W = np.array(W[:10])
if len(W)==10:
    W = np.array(W)
    plt.plot(epoch_nums, W)
    plt.xlabel("No. of Iteration")
    plt.ylabel("loss - lossa after updata")
    plt.title("")
    plt.show()





