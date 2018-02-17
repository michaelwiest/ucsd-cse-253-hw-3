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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

def imshow(inp, weight_t=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    if weight_t:
        inp /= (inp.max())
        inp = inp + abs(inp.min())
    plt.imshow(inp)

vgg = models.vgg16(pretrained=True)
for param in vgg.parameters():
    # make all parameters untrainiable except last
    param.requires_gradient = False

layer1_model = nn.Sequential(*list(vgg.features.children())[0:1])
for param in layer1_model.parameters():
    param.requires_gradient=False
layer5_model = nn.Sequential(*list(vgg.features.children())[0:29])
for param in layer5_model.parameters():
    param.requires_gradient=False

iterate = 0
for i, data in enumerate(train_data, 0):
    inputs, labels = data
    labels = labels.long()
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs_1 = layer1_model(inputs)
    outputs_2 = layer5_model(inputs)
    outputs_2 = outputs_2.data
    outputs_1 = outputs_1.data

    inputs = inputs.data
    for index in [1,5,8,10,11]:
        fig1 = plt.figure()
        imshow(inputs[index])
        plt.axis('off')
        plt.title('Original Image')
        fig1.savefig('/Users/jenniferdawkins/Desktop/pics5/first_layer_' +str(index) + '.png')

        fig2 = plt.figure()
        for jj in range(20):
            plt.subplot(4,5,jj+1)
            image_F = outputs_1[index,jj,:,:]
            plt.imshow(image_F.numpy())
            plt.axis('off')
        fig2.savefig('/Users/jenniferdawkins/Desktop/pics5/second_layer_' +str(index) + '.png')

        fig3 = plt.figure()
        for kk in range(20):
            plt.subplot(4,5,kk+1)
            image_F = outputs_2[index,kk,:,:]
            plt.imshow(image_F.numpy())
            plt.axis('off')
        fig3.savefig('/Users/jenniferdawkins/Desktop/pics5/third_layer_' +str(index) + '.png')

    iterate = iterate+1
    if iterate==1:
        break

mm = vgg.double()
filters = mm.modules
body_model = [i for i in mm.children()][0]
layer1 = body_model[0]
tensor = layer1.weight.data


fig_w = plt.figure()
for filter in range(tensor.shape[0]):
    plt.subplot(8,8,filter+1)
    imshow(tensor[filter],weight_t = True)
    plt.axis('off')
fig_w.savefig('/Users/jenniferdawkins/Desktop/pics5/weights_3.png')

fig_color = plt.figure()
imshow(tensor[filter],weight_t = True)
plt.colorbar()
fig_color.savefig('/Users/jenniferdawkins/Desktop/pics5/colorbar.png')
