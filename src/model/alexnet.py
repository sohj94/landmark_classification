from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, AdaptiveAvgPool2d
import torch.nn.functional as F

class alexnet(nn.Module) :
    def __init__(self) :
        super(alexnet, self).__init__()
        self.conv1 = Conv2d(3, 96, (11,11), (4,4), (0,0))
        self.conv2 = Conv2d(96, 256, (5,5), (1,1), (2,2))
        self.conv3 = Conv2d(256, 384, (3,3), (1,1), (1,1))
        self.conv4 = Conv2d(384, 384, (3,3), (1,1), (1,1))
        self.conv5 = Conv2d(384, 256, (3,3), (1,1), (1,1))
        self.fc1 = Linear(256, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, 1049)
        self.mp1 = MaxPool2d((3,3), (2,2))
        self.mp2 = MaxPool2d((3,3), (2,2))
        self.mp3 = MaxPool2d((3,3), (2,2))

    def forward(self, x) :
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.mp1(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.mp2(x)
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = self.mp3(x)
        print(x.shape)
        x = AdaptiveAvgPool2d(1)(x).squeeze()
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x
