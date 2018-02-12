from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def get_accuracy(dataloader, net, classes):
    correct = 0
    total = 0
    for data in dataloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100.0 * correct / total

def get_class_correct(dataloader, net, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in dataloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_perc = []
    # for i in range(10):
        # print('Accuracy of %5s : %2d %%' % (
        #     classes[i], 100 * class_correct[i] / class_total[i]))


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 6, 5)
        # self.conv3 = nn.Conv2d(6, 1, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
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

# trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR-10', train=True,
#                                         transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=25,
#                                           shuffle=True, num_workers=2)

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



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('Defined Everything')


# net.cuda()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
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
        if i % 50 == 0:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            print('test accuracy:\n')
            print(get_accuracy(testloader, net, classes))

            print('validation accuracy:\n')
            print(get_accuracy(validationloader, net, classes))
    print('Completed an Epoch')

print('test accuracy:\n')
print(get_accuracy(testloader, net, classes))

print('validation accuracy:\n')
print(get_accuracy(validationloader, net, classes))

#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
