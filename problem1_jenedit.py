from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('Defined Everything')


# net.cuda()
rl_vec=[]
rl_epoch=[]
rl_epoch_test=[]
acc_test=[]
acc_train=[]
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    running_loss_test = 0.0
    total_train=0.0
    correct_train=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        rl_vec.append(running_loss)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    correct = 0
    total = 0
    for data_test in testloader:
        images, labels_test = data
        outputs_test = net(Variable(images))
        _, predicted_test = torch.max(outputs_test.data, 1)
        total += labels_test.size(0)
        correct += (predicted_test == labels_test).sum()
    acc = (100 * float(correct) / total)
    if acc == 0:
        pdb.set_trace()

    print('Accuracy of the network on the test images:' + str(
        100 * correct / total))
    print('Completed an Epoch')
    rl_epoch.append(rl_vec[-1])
    # rl_epoch_test.append(1-(acc/100.0))
    acc_test.append(acc)
    # acc_train.append(100*(1-rl_vec[-1]))
print('Test Accuracy:' + str(acc_test))
print('Train Loss: ' + str(rl_epoch))
# plt.figure()
# plt.plot(range(2),acc_train,'Training Accuracy per Epoch')
# plt.plot(range(2),acc_test,'Testing Accuracy per Epoch')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
#
# plt.show()
#
# plt.figure()
# plt.plot(range(2),rl_epoch,'Training Loss per Epoch')
# plt.plot(range(2),rl_epoch_train,'Testing Loss per Epoch')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
#
# plt.show()
