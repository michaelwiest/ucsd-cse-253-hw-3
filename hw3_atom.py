import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#torch.utils.data.sampler.SubsetRandomSampler(indices)

#input, conv, pool, relu, conv, pool, relu, fc, fc
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #the input is 3 RBM, the output is 32 because we want 32 filters 3x3
        self.conv1 = nn.Conv2d(3, 32, 5)
        #the input is the previous 32 filters and output is 64 filters, with 3x3
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.mp = nn.MaxPool2d(2, stride=2) #2X2 with stride 2
        self.fc = nn.Linear(1600, 120) #fully connected
        self.fc2 = nn.Linear(120, 10) #fully connected same number neurons as classes 10

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x))) #conv then mp then relu
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) #flatten for fc
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x)


net = Net()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #stochastic gradient descent
optimizer = optim.RMSprop(net.parameters(), lr = 0.0001)
#atom-- LR

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #model output
        loss = criterion(outputs, labels) #cross entropy
        loss.backward()
        optimizer.step() #stochastic gradient descent

        # print statistics
        running_loss += loss.data[0] #combine loss in minibatch
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % #print epoch and then iteration
                  (epoch + 1, i + 1, running_loss / 2000)) #loss
            running_loss = 0.0

print('Finished Training')
#cross validation 
