from caltech import Caltech256
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pdb
import torchvision
import torchvision.transforms as transforms
import torch

use_gpu = torch.cuda.is_available()

vgg = models.vgg16(pretrained=True)
for param in vgg.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False
features_in = vgg.classifier._modules['6'].in_features

softmax_model = nn.Sequential(nn.Linear(features_in,256),nn.Softmax())
# vgg.classifier._modules['6'] = softmax_model
vgg.fc = softmax_model
if use_gpu:
    vgg.cuda()

example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

caltech256_train = Caltech256("256_ObjectCategories",
                    example_transform, train=True)

train_data = torch.utils.data.DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
rl_epoch=[]
rl_epoch_test=[]
acc_test=[]
acc_train_vec=[]
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    total_train = 0.0
    correct_train = 0.0
    for i, data in enumerate(train_data, 0):
        # get the inputs
        inputs, labels = data

        labels_orig = labels.long()
        labels = labels.long()
        # wrap them in Variable
        if use_gpu:
            labels_orig = labels_orig.cuda()
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels_orig.size(0)
        correct_train += (predicted_train == labels_orig).sum()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('Completed 2000 Minibatches')
            print('Running Loss'+str(running_loss))
            print('Accuracy' + str((100 * float(correct_train) / total_train)))
            #rl_vec.append(running_loss)
            #running_loss = 0.0
    acc_train = (100 * float(correct_train) / total_train)

    correct = 0.0
    total = 0.0
    running_loss_test = 0.0
    for data_test in test_data:
        images, labels_test = data
        labels_test = labels_test.long()
        if use_gpu:
            labels_test_orig = labels_test.cuda()
            images, labels_test = Variable(images.cuda()),Variable(labels_test.cuda())
        else:
            labels_test_orig = labels_test
            images, labels_test = Variable(images),Variable(labels_test)


        outputs_test = vgg(images)
        _, predicted_test = torch.max(outputs_test.data, 1)

        loss_test = criterion(outputs_test, torch.max(labels_test, 1)[1])
        running_loss_test += loss_test.data[0]

        total += labels_test_orig.size(0)
        correct += (predicted_test == labels_test_orig).sum()
    acc = (100 * float(correct) / total)

    print('Accuracy of the network on the test images:' + str(
        100 * correct / total))
    print('Completed an Epoch')
    rl_epoch.append(running_loss / float(total_train))
    rl_epoch_test.append(running_loss_test / float(total))
    acc_test.append(acc)
    acc_train_vec.append(acc_train)

    fh = open('test_acc.txt', 'a')
    fh.write(str(acc_test[-1]))
    fh.close

    fh = open('test_loss.txt', 'a')
    fh.write(str(rl_epoch_test[-1]))
    fh.close

    fh = open('train_acc.txt', 'a')
    fh.write(str(acc_train_vec[-1]))
    fh.close

    fh = open('train_loss.txt', 'a')
    fh.write(str(rl_epoch[-1]))
    fh.close
print('Test Accuracy:' + str(acc_test))
print('Train Loss: ' + str(rl_epoch))
