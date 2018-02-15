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

def get_accuracy(dataloader, net):
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        labels = labels.long()
#         labels = labels - 1
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100.0 * float(correct) / total

def get_class_accuracy(dataloader, net, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in dataloader:
        inputs, labels = data
        labels = labels.long()
#         labels = labels - 1
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_perc = []

# use_gpu = torch.cuda.is_available()
vgg = models.vgg16(pretrained=True)
for param in vgg.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False
features_in = vgg.classifier._modules['6'].in_features

softmax_model = nn.Sequential(nn.Linear(features_in,256),nn.Softmax())
vgg.classifier._modules['6'] = softmax_model
vgg.classifier._modules['6'].reguires_gradient = True
# if use_gpu:
vgg.cuda()

ex_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/",
                   ex_transform, train=True)

caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/",
                   ex_transform, train=False)

train_data = torch.utils.data.DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4)

test_data = torch.utils.data.DataLoader(
    dataset = caltech256_test,
    batch_size=8,
    shuffle=False,
    num_workers=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.0001, momentum=0.9)
loss_train_vec = []
loss_test_vec = []
acc_train=[]
acc_test=[]
epochs = 10
running_loss = 0.0
running_loss_test = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times

    for i, data in enumerate(train_data, 0):
        # get the inputs
        inputs, labels = data

        labels = labels.long()
#         if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         else:
#             inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg(inputs)
        labels = labels - 1
        labels = labels.squeeze(1)
        loss = criterion(outputs, labels)
#         loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = running_loss + loss.data[0]
#         print('Batch Loss: ' + str(loss.data[0]))
    loss_train_vec.append(running_loss)
    print loss.data[0]
    acc_train.append(get_accuracy(train_data,vgg))
    for i, data, in enumerate(test_data):
        inputs_t, labels_t = data
        labels_t = labels_t.long()
        labels_t = labels_t - 1

#         if use_gpu:
        inputs_t, labels_t = Variable(inputs_t.cuda()), Variable(labels_t.cuda())
#         else:
#             inputs_t, labels_t = Variable(inputs_t), Variable(labels_t)

        outputs_test = vgg(inputs_t)
        labels_t = labels_t.squeeze(1)
        loss_test = criterion(outputs_test, labels_t)
#         pdb.set_trace()
#         loss_test = criterion(outputs,labels_t)

        running_loss_test = running_loss_test + loss_test.data[0]
#     pdb.set_trace()
    loss_test_vec.append(running_loss_test)
    acc_test.append(get_accuracy(test_data,vgg))
    print(str(i))
    print('Running Loss: '+str(running_loss))
    print('Completed an Epoch')
    print('Test Accuracy: ' + str(acc_test[-1]))
    print('Test Loss' + str(loss_test_vec[-1]))
    running_loss = 0.0
    running_loss_test = 0.0

    fh = open('test_acc_4.txt', 'a')
    fh.write(str(acc_test[-1])+ ', ')
    fh.close

    fh = open('test_loss_4.txt', 'a')
    fh.write(str(loss_test_vec[-1])+ ', ')
    fh.close

    fh = open('train_acc_4.txt', 'a')
    fh.write(str(acc_train[-1])+ ', ')
    fh.close

    fh = open('train_loss_4.txt', 'a')
    fh.write(str(loss_train_vec[-1])+ ', ')
    fh.close

# Total loss
import matplotlib.pyplot as plt
plt.semilogy(range(epochs), loss_train_vec, label='Train loss')
plt.semilogy(range(epochs), loss_test_vec, label='Test loss')
# plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over \n{} Epochs'.format(epochs), fontsize=16)
plt.legend(loc='upper right')
plt.show()

# Total accuracy
plt.plot(range(epochs), acc_train, label='Train accuracy')
plt.plot(range(epochs), acc_test, label='Test accuracy')
# plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy')
plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
plt.legend(loc='lower right')
plt.show()
