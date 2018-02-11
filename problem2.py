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


vgg = models.vgg16_bn(pretrained=True)
for param in vgg.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False
features_in = vgg.classifier._modules['6'].in_features

softmax_model = nn.Sequential(nn.Linear(features_in,256),nn.Softmax())
vgg.classifier._modules['6'] = softmax_model
# vgg.cuda()

example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
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

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print('Completed an Epoch')
