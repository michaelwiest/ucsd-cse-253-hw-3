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
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


caltech256_train = Caltech256("256_ObjectCategories",
                    example_transform, train=True)

train_data = torch.utils.data.DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4)

image = caltech256_train.__getitem__(1)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

for i, data in enumerate(train_data, 0):
    images, labels = data
    imshow(images[1])


vgg = models.vgg16(pretrained=True)
for param in vgg.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False
features_in = vgg.classifier._modules['6'].in_features

softmax_model = nn.Sequential(nn.Linear(features_in,256),nn.Softmax())
vgg.classifier._modules['6'] = softmax_model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
rl_epoch=[]
running_loss = 0.0

layer1_model = nn.Sequential(*list(vgg.features.children())[0:2])
for param in layer1_model.parameters():
    param.requires_gradient=False
layer5_model = nn.Sequential(*list(vgg.features.children())[:-3])
for param in layer5_model.parameters():
    param.requires_gradient=False

iterate = 0
for i, data in enumerate(train_data, 0):
    inputs, labels = data
    for index in [1,5,8,10,11]:
        imshow(inputs[index])

        labels = labels.long()
        if use_gpu:
            inputs, labels = Variable(inputs[1].cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs_1 = layer1_model(inputs)
        outputs_2 = layer5_model(inputs)
        outputs_2 = outputs_2.data
        outputs_1 = outputs_1.data

        inputs = inputs.data
        for index in [1,5,8,10,11]:
            fig1 = plt.figure()
            imshow(inputs[index])
            plt.title('Original Image')
            fig1.savefig('/Users/jenniferdawkins/Desktop/pics4/first_layer_' +str(index) + '.png')

            fig2 = plt.figure()
            for jj in range(20):
                plt.subplot(4,5,jj+1)
                image_F = outputs_1[index,jj,:,:]
                plt.imshow(image_F.numpy())
            # plt.title('First 20 (of 64) Filters of First Conv Layer')
            fig2.savefig('/Users/jenniferdawkins/Desktop/pics4/second_layer_' +str(index) + '.png')

            # imshow(outputs_1[index])
            # plt.title('First Filter')
            # plt.show()

            fig3 = plt.figure()
            for kk in range(20):
                plt.subplot(4,5,kk+1)
                image_F = outputs_2[index,kk,:,:]
                plt.imshow(image_F.numpy())
            # plt.title('First 20 (of 512) Filters of Last Conv Layer')
            fig3.savefig('/Users/jenniferdawkins/Desktop/pics4/third_layer_' +str(index) + '.png')
            # plt.show()
        pdb.set_trace()
    iterate = iterate+1
    if iterate==1:
        break
