from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
import torch.nn.functional as F

class three_layer_conv_net(nn.Module) :
    def __init__(self) :
        super(Network, self).__init__()
        self.conv1 = Conv2d(3, 64, (3,3), (1,1), (1,1))
        self.conv2 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.conv3 = Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.fc = Linear(64, 1049)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x
