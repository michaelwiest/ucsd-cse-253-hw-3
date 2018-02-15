from __future__ import print_function
from data_loader import *
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainloader, validationloader = get_train_valid_loader(data_dir='/datasets/CIFAR-10',
                                                       batch_size=25,
                                                       augment=False,
                                                       random_seed=2)

testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=25,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

#CNN network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #convolutional layers
        #the input is 3 RBM, the output is 64 because we want 64 filters 3x3
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        #the input is the previous 64 filters and output is 64 filters, with 3x3
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)

        #batch normalization
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)

        #max pooling
        self.mp = nn.MaxPool2d(2, stride=2) #2X2 with stride 2

        self.fc = nn.Linear(256*4*4, 500) #fully connected
        self.fc2 = nn.Linear(500, 10) #fully connected with classes 10

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.mp(F.relu(self.batchnorm1(self.conv2(x))))

        x = F.relu(self.batchnorm2(self.conv3(x)))
        x = self.mp(F.relu(self.batchnorm2(self.conv4(x))))

        x = F.relu(self.batchnorm3(self.conv5(x)))
        x = self.mp(F.relu(self.batchnorm3(self.conv6(x))))

        x = x.view(x.size(0), -1) #flatten for fc
        #print (x.size(1))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x) #softmax classifier

net = Net()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #stochastic gradient descent
#optimizer = optim.RMSprop(net.parameters(), lr = 0.0001)
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.cuda()

train_accuracy = []
test_accuracy = []
validation_accuracy = []

train_class_accuracy = []
test_class_accuracy = []
validation_class_accuracy = []


epochs = 80

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #model output
        loss = criterion(outputs, labels) #cross entropy
        loss.backward()
        optimizer.step() #stochastic gradient descent

        # print statistics
        running_loss += loss.data[0] #combine loss in minibatch
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % #print epoch and then iteration
                  (epoch + 1, i + 1, running_loss / 2000)) #loss
            running_loss = 0.0
    print('Completed an Epoch')

    train_accuracy.append(get_accuracy(trainloader, net, classes))
    test_accuracy.append(get_accuracy(testloader, net, classes))
    validation_accuracy.append(get_accuracy(validationloader, net, classes))

    train_class_accuracy.append(get_class_accuracy(trainloader, net, classes))
    test_class_accuracy.append(get_class_accuracy(testloader, net, classes))
    validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes))

#accuracy plots
plt.plot(range(epochs), train_accuracy, label='Train accuracy')
plt.plot(range(epochs), test_accuracy, label='Test accuracy')
plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.legend(loc='upper right')
plt.style.use('ggplot')
plt.show()

#accuracy per class graphs
f, axarr = plt.subplots(2, 5)
for i in range(len(classes)):
    if int((i) / 5) > 0:
        row = 1
        col = i % 5
    else:
        row = 0
        col = i

    print(row, col)
    axarr[row, col].plot(range(epochs), list(np.array(train_class_accuracy)[:, i]), label='Train accuracy')
    axarr[row, col].plot(range(epochs), list(np.array(test_class_accuracy)[:, i]), label='Test accuracy')
    axarr[row, col].plot(range(epochs), list(np.array(validation_class_accuracy)[:, i]), label='Validation accuracy')
    axarr[row, col].set_title('Accuracy for class:\n{}'.format(classes[i]))

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.show()
