from __future__ import print_function
from data_loader import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




def get_accuracy(dataloader, net, classes):
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100.0 * correct / total

def get_class_accuracy(dataloader, net, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_perc = []

    for i in range(10):
        class_perc.append(100.0 * class_correct[i] / class_total[i])
    return class_perc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #the input is 3 RBM, the output is 32 because we want 32 filters 3x3
        self.conv0 = nn.Conv2d(3, 32, 5) # output is 28
        self.conv1 = nn.Conv2d(32, 64, 5) # Output is 24
        self.mp = nn.MaxPool2d(2, stride=2) # Output is 12
        self.conv2 = nn.Conv2d(64, 128, 3, padding=2) # output is 14
        self.conv3 = nn.Conv2d(128, 64, 1) # Output is 14.
        # Do another map. Output is 7

        self.fc0 = nn.Linear(7 * 7 * 64, 120) #fully connected
        self.fc1 = nn.Linear(120, 120)
        self.fc2 = nn.Linear(120, 10) #fully connected same number neurons as classes 10

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.mp(x)
        in_size = x.size(0)
        x = x.view(in_size, -1) #flatten for fc
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)




'''
Load in all of the data. And set the transforms.
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 5
trainloader, validationloader = get_train_valid_loader(data_dir='/datasets/CIFAR-10',
                                                       batch_size=batch_size,
                                                       augment=False,
                                                       random_seed=2)


testloader = get_test_loader(data_dir='/datasets/CIFAR-10',
                             batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




'''
Instantiate net and optimizer.
'''


net = Net()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)



net.cuda()

'''
Training.

'''

train_accuracy = []
test_accuracy = []
validation_accuracy = []

train_class_accuracy = []
test_class_accuracy = []
validation_class_accuracy = []

epochs = 4
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Completed an Epoch')
    train_accuracy.append(get_accuracy(trainloader, net, classes))
    test_accuracy.append(get_accuracy(testloader, net, classes))
    validation_accuracy.append(get_accuracy(validationloader, net, classes))

    train_class_accuracy.append(get_class_accuracy(trainloader, net, classes))
    test_class_accuracy.append(get_class_accuracy(testloader, net, classes))
    validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes))


plt.plot(range(epochs), train_accuracy, label='Train accuracy')
plt.plot(range(epochs), test_accuracy, label='Test accuracy')
plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.legend(loc='upper right')
plt.show()


# Can do something similar for accuracy for each class.
