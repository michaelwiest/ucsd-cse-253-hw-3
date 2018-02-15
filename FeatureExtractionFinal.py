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
from torch.optim import lr_scheduler
use_gpu = torch.cuda.is_available()

vgg2 = models.vgg16(pretrained=True)
vgg2.cuda()
for param in vgg2.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False

size_dict={1:28*28*256,2:14*14*512}
layer3_model = nn.Sequential(*list(vgg2.features.children())[0:17])
final_layer_3 = nn.Sequential(nn.Linear(size_dict[1],1024),nn.ReLU(),nn.Linear(1024,256))
final_layer_3.cuda()

layer4_model = nn.Sequential(*list(vgg2.features.children())[0:24])
final_layer_4 = nn.Sequential(nn.Linear(size_dict[2],1024),nn.ReLU(),nn.Linear(1024,256))
final_layer_4.cuda()

model_dict = {1:layer3_model,2:layer4_model}
final_layer_dict = {1:final_layer_3,2:final_layer_4}


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/",
                   data_transforms['train'], train=True)

caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/",
                   data_transforms['val'], train=False)

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
data_dict = {}
for key, vgg_model in model_dict.items():
    vgg_model.cuda()
    optimizer = optim.SGD(final_layer_dict[key].parameters(), lr=0.0001, momentum = .9,
                      weight_decay = .4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_train_vec = []
    loss_test_vec = []
    acc_train=[]
    acc_test=[]
    epochs = 20
    running_loss = 0.0
    running_loss_test = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        total=0.0
        correct = 0.0
        total_t = 0.0
        correct_t = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data

            labels = labels.long()
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = vgg_model(inputs)
            outputs = outputs.view(-1,size_dict[key])
            outputs = final_layer_dict[key](outputs)
            labels = labels - 1
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            running_loss = running_loss + loss.data[0]
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        loss_train_vec.append(running_loss / float(total))
        acc_train.append(100.0 * float(correct) / total)
        for i, data, in enumerate(test_data):
            inputs_t, labels_t = data
            labels_t = labels_t.long()
            labels_t = labels_t - 1
            inputs_t, labels_t = Variable(inputs_t.cuda()), Variable(labels_t.cuda())
            outputs_t = vgg_model(inputs_t)
            outputs_t = outputs_t.view(-1,size_dict[key])
            outputs_test = final_layer_dict[key](outputs_t)

            labels_t = labels_t.squeeze(1)
            loss_test = criterion(outputs_test, labels_t)
            _, predicted_t = torch.max(outputs_test.data, 1)

            total_t += labels_t.size(0)
            correct_t += (predicted_t == labels_t.data).sum()
            running_loss_test = running_loss_test + loss_test.data[0]
        loss_test_vec.append(running_loss_test / float(total_t))
        acc_test.append(100.0 * float(correct_t) / total_t)
        print(str(i))
        print('Completed Epoch' + str(epoch))
        print('Test Accuracy: ' + str(acc_test[-1]))
        print('Test Loss' + str(loss_test_vec[-1]))
        running_loss = 0.0
        running_loss_test = 0.0

        fh = open('Model' + str(key) + 'test_acc_4.txt', 'a')
        fh.write(str(acc_test[-1])+ ', ')
        fh.close

        fh = open('Model' + str(key) + 'test_loss_4.txt', 'a')
        fh.write(str(loss_test_vec[-1])+ ', ')
        fh.close

        fh = open('Model' + str(key) + 'train_acc_4.txt', 'a')
        fh.write(str(acc_train[-1])+ ', ')
        fh.close

        fh = open('Model' + str(key) + 'train_loss_4.txt', 'a')
        fh.write(str(loss_train_vec[-1])+ ', ')
        fh.close
    data_dict['trainloss_'+str(key)] = running_loss
    data_dict['testloss_'+str(key)] = running_loss_test
    data_dict['trainacc_'+str(key)] = acc_train
    data_dict['testacc_'+str(key)] = acc_test

    print('One model complete, onto next')
# Total loss

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.semilogy(range(epochs), data_dict['trainloss_'+str(2)], label='Train Loss')
plt.semilogy(range(epochs), data_dict['testloss_'+str(2)], label='Test Loss')
plt.text(5,.15,'Min Test Loss = ' + str(round(min(loss_test_vec),3)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('4 Convolution Layer Loss over \n{} Epochs'.format(epochs), fontsize=16)
plt.legend(loc='upper right')
plt.show()

# Total loss
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.semilogy(range(epochs), data_dict['trainloss_'+str(1)], label='Train Loss')
plt.semilogy(range(epochs), data_dict['testloss_'+str(1)], label='Test Loss')
plt.text(5,.15,'Min Test Loss = ' + str(round(min(loss_test_vec),3)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('3 Convolution Layer Loss over \n{} Epochs'.format(epochs), fontsize=16)
plt.legend(loc='upper right')
plt.show()

# Total accuracy
plt.plot(range(epochs), data_dict['trainacc_'+str(2)], label='Train accuracy')
plt.plot(range(epochs), data_dict['testacc_'+str(2)], label='Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy (%)')
plt.title('4 Convolution Layer Accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
plt.text(5,30,'Max Test Accuracy = ' + str(round(max(acc_test),3)))
plt.legend(loc='lower right')
plt.show()

# Total accuracy
plt.plot(range(epochs), data_dict['trainacc_'+str(1)], label='Train accuracy')
plt.plot(range(epochs), data_dict['testacc_'+str(1)], label='Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy (%)')
plt.title('3 Convolution Layer Accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
plt.text(5,20,'Max Test Accuracy = ' + str(round(max(acc_test),3)))
plt.legend(loc='lower right')
plt.show()
