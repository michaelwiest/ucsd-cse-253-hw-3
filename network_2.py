#######
# Developed by Pablo Tostado Marcos
# Last modified: Feb 15th 2018
#######


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
from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import random

######################## FUNCTIONS ########################

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

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


######################## IMPORT DATA ####################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

global batch_size
batch_size = 20

# Load Training + Validation
trainloader, validationloader = get_train_valid_loader(data_dir='/datasets/CIFAR-10',
                                                       batch_size=batch_size,
                                                       augment=False,
                                                       random_seed=2)
# Load Testing
testloader = get_test_loader(data_dir='/datasets/CIFAR-10',
                             batch_size=batch_size,
                             pin_memory=True)




classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


###
# Trsaining: 1800 batches of 25 images (45000 images)
# Validation: 200 batches of 25 images (5000 images)
# Testing: 400 batches of 25 images (10000 images)
###


####################### VISUALIZE IMAGES ############################################

# trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=False,
#                                        transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=25,
#                                          shuffle=False, num_workers=2)

# ##### Print figure with 1 random image from each class
# train_labels = [] # Get labels
# for im in xrange(0, len(trainset)):
#     train_labels.append(trainset[im][1])
# train_labels = np.array(train_labels)

# fig = plt.figure(figsize=(8,3))
# for i in range(10):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     idx = np.where(train_labels==i)     # Find images with target label
#     idx = random.choice(idx[0])         # Pick random idx of current class
#     img = trainset[idx][0] #Take image
#     ax.set_title(classes[i])
#     imshow(img)
# plt.show(block = False)

######################## NET CLASS Example ########################

# conv2(in_channels, out_channels, kernel, stride, padding)
# MaxPool2d(kernel, stride, padding)

class Net_ex(nn.Module):
    def __init__(self):
        super(Net, self).__init__()            # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 6, 5)        # output = 6x14x14
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)       # x = 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # x = 1x120
        self.fc2 = nn.Linear(120, 84)          # x = 1x84
        self.fc3 = nn.Linear(84, 10)           # x = 1x10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # tensor.view -> returns a new tensor with the same data, diff dimensions.
        # If param = -1, dimension inferred from other dimensions.
        x = x.view(-1, 16 * 5 * 5)             # x = 400
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


######################## NET CLASS 1 ########################
# https://arxiv.org/pdf/1608.06037.pdf

# conv2(in_channels, out_channels, kernel, stride, padding)
# MaxPool2d(kernel, stride, padding)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()            # Input: 3x32x32

        self.conv1 = nn.Conv2d(3, 64, 3, padding=2)        # output = 64x34x34
        self.conv2 = nn.Conv2d(64, 128, 3, padding=2)        # output = 128x36x36
        self.conv3 = nn.Conv2d(128, 128, 3, padding=2)        # output(after pool) = 128x19x19
        self.conv5 = nn.Conv2d(128, 64, 5, padding=2)       # x = 64x6x6
        self.conv1x1 = nn.Conv2d(64, 1, 1)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # x = 1x64
        self.fc2 = nn.Linear(128, 256)          # x = 1x128
        self.fc3 = nn.Linear(256, 10)           # x = 1x10

        self.pool = nn.MaxPool2d(2, stride=2)
        self.avgPool = nn.AvgPool2d(2, stride=2)

        self.batch10 = nn.BatchNorm1d(10)
        self.batch64 = nn.BatchNorm1d(64)
        self.batch128 = nn.BatchNorm1d(128)
        self.batch256 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.conv1(x)  #3->64
        x = self.batch64(x)
        x = F.relu(x)

        x = self.conv2(x)  #64->128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x)  #128->128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x) #128->128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x) #128 + pool
        x = self.batch128(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x) #128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x) #128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x) #128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.conv3(x) #128 + Average
        x = self.batch128(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)   #64
        x = self.batch64(x)
        x = F.relu(x)

#         x = self.conv1x1(x)   # 1x1 convolution. Works worse...

        x_size = x.size(0)  #Batch_size (check size to design fc1 accordingly)
        x = x.view(x_size, -1)
#         print(x.size())

        x = self.fc1(x)      #fc1: 128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.fc2(x)      #fc2: 256
        x = self.batch256(x)
        x = F.relu(x)

        x = self.fc3(x)      #fc3: 10
        x = self.batch10(x)
        x = F.relu(x)

        return F.log_softmax(x)


######################## NET CLASS 2 ########################

# 3 branches with different filters that come together at the end for a final softmax

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=2)

        self.conv11 = nn.Conv2d(64, 128, 2, padding=2)
        self.conv21 = nn.Conv2d(64, 128, 3, padding=2)
        self.conv31 = nn.Conv2d(64, 128, 5, padding=2)

        self.conv12 = nn.Conv2d(128, 128, 2, padding=2)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=2)
        self.conv32 = nn.Conv2d(128, 128, 5, padding=2)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.avgPool = nn.AvgPool2d(2, stride=2)

        self.batch10 = nn.BatchNorm1d(10)
        self.batch64 = nn.BatchNorm1d(64)
        self.batch128 = nn.BatchNorm1d(128)
        self.batch256 = nn.BatchNorm1d(256)

#         self.conv1x1 = nn.Conv2d(128, 1, 1)

        self.fc1 = nn.Linear(62592, 128)  # After Concatenation: x = 1x128
        self.fc2 = nn.Linear(128, 256)          # x = 1x256
        self.fc3 = nn.Linear(256, 10)           # x = 1x10


    def forward(self, x):

        x = self.conv1(x)  # 3->64
        x = self.batch64(x)
        x = F.relu(x)

        x1 = F.relu(self.batch128(self.conv11(x)))   # 64->128
        x2 = F.relu(self.batch128(self.conv21(x)))
        x3 = F.relu(self.batch128(self.conv31(x)))

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = self.pool(x1) #Pool
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = F.relu(self.batch128(self.conv12(x1)))  #128->128
        x2 = F.relu(self.batch128(self.conv22(x2)))
        x3 = F.relu(self.batch128(self.conv32(x3)))

        x1 = self.pool(x1) #Average Pool
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1_size = x1.size(0)  #Batch_size (check size to design fc1 accordingly)
        x1 = x1.view(x1_size, -1)
#         print(x1.size())

        x2_size = x2.size(0)  #Batch_size (check size to design fc1 accordingly)
        x2 = x2.view(x2_size, -1)
#         print(x2.size())

        x3_size = x3.size(0)  #Batch_size (check size to design fc1 accordingly)
        x3 = x3.view(x3_size, -1)
#         print(x3.size())

        x = torch.cat((x1,x2,x3),1)
#         print(x.size())


        x = self.fc1(x)      #fc1: 128
        x = self.batch128(x)
        x = F.relu(x)

        x = self.fc2(x)      #fc2: 256
        x = self.batch256(x)
        x = F.relu(x)

        x = self.fc3(x)      #fc3: 10
        x = self.batch10(x)
        x = F.relu(x)

        return F.log_softmax(x)

####################### RUN NET ###################################

net = Net1()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
print('Defined Everything')


train_accuracy = []
test_accuracy = []
validation_accuracy = []

train_class_accuracy = []
test_class_accuracy = []
validation_class_accuracy = []


epochs = 15
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

print('test accuracy:\n')
print(get_accuracy(testloader, net, classes))

print('validation accuracy:\n')
print(get_accuracy(validationloader, net, classes))

'''
Plotting
'''

plt.style.use('ggplot')

'''
Total accuracy
'''
plt.figure()
plt.plot(range(epochs), train_accuracy, label='Train accuracy')
plt.plot(range(epochs), test_accuracy, label='Test accuracy')
plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy')
plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
plt.legend(loc='lower right')
plt.show(block=False)



'''
Accuracy by class.
'''

f, axarr = plt.subplots(2, 5, figsize=(18,9))
for i in range(len(classes)):
    if int((i) / 5) > 0:
        row = 1
        col = i % 5
    else:
        row = 0
        col = i

    print(row, col)
    axarr[row, col].plot(range(len(train_class_accuracy)), list(np.array(train_class_accuracy)[:, i]), label='Train accuracy')
    axarr[row, col].plot(range(len(test_class_accuracy)), list(np.array(test_class_accuracy)[:, i]), label='Test accuracy')
    axarr[row, col].plot(range(len(validation_class_accuracy)), list(np.array(validation_class_accuracy)[:, i]), label='Validation accuracy')
    axarr[row, col].set_title('Accuracy for\nclass: {}'.format(classes[i]))

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.suptitle('Accuracy By Class over {} Epochs'.format(len(train_accuracy)), fontsize=16)
plt.figlegend(loc = 'lower center', ncol=5, labelspacing=0. )
plt.show()
