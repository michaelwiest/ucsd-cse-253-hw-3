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
import csv

plt.style.use('ggplot')


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
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3, 32, 4, stride=2, padding=2), # output is 17
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, padding=3), # Output is 20
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2), # Output is 10
                    nn.Conv2d(64, 128, 3, padding=2), # output is 12
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout2d(),
                    nn.Conv2d(128, 32, 1), # Output is 12
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2), # Output is 6
                    nn.Conv2d(32, 64, 2, padding=1), #Output is 7
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=2, stride=2), #Output is 5
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 32, 1), # Output is 5
                    nn.BatchNorm1d(32),
                    nn.ReLU()
                    ),

                nn.Sequential(
                    nn.Conv2d(3, 32, 5, padding=1, dilation=1), # output is 30
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=3, padding=2, dilation=1), # Output is 10
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2) # Output is 5
                    )
                    ])
            self.classifier = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(32 * 5 * 5, 128), #fully connected
    #                 nn.Dropout(),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                ),
                nn.Sequential(
                    nn.Linear(64 * 5 * 5, 128), #fully connected
    #                 nn.Dropout(),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            ])

            self.final_fc = nn.Sequential(
                nn.Linear(40, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 10)
            )


        def forward(self, x):
            outs = [0] * len(self.branches)
            for i in range(len(self.branches)):
                outs[i] = self.branches[i](x)
                outs[i] = outs[i].view(outs[i].size(0), -1)
                outs[i] = self.classifier[i](outs[i])
            cat = torch.cat([out for out in outs], 1)
            # out = cat.view(cat.size(0), -1)
            # out = self.final_fc(out)
            return F.log_softmax(cat)


'''
Load in all of the data. And set the transforms.
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

global batch_size
batch_size = 20
trainloader, validationloader = get_train_valid_loader(data_dir='/datasets/CIFAR-10',
                                                       batch_size=batch_size,
                                                       augment=False,
                                                       random_seed=2,
                                                       pin_memory=True)


testloader = get_test_loader(data_dir='/datasets/CIFAR-10',
                             batch_size=batch_size,
                             pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




'''
Instantiate net and optimizer.
'''


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.03)



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

epochs = 30
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




'''

Plotting

'''

'''
Total accuracy
'''
fig=plt.figure(figsize=(16, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(range(len(train_accuracy)), train_accuracy, label='Train accuracy: {}%'.format(str(train_accuracy[-1])[:4]))
plt.plot(range(len(test_accuracy)), test_accuracy, label='Test accuracy: {}%'.format(str(test_accuracy[-1])[:4]))
plt.plot(range(len(validation_accuracy)), validation_accuracy, label='Validation accuracy: {}%'.format(str(validation_accuracy[-1])[:4]))
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy')
plt.title('Training accuracy over: \n{} Iterations'.format(len(train_accuracy)), fontsize=16)
plt.legend(loc='lower right')
plt.show()

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

with open('log.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(train_accuracy)
    writer.writerow(test_accuracy)
    writer.writerow(validation_accuracy)

with open('log_class.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(train_class_accuracy)
    writer.writerow(test_class_accuracy)
    writer.writerow(validation_class_accuracy)
