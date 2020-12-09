import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, AdaptiveAvgPool2d
import torch.nn.functional as F

class alexnet(nn.Module) :
    def __init__(self) :
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1049),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x    
    # def __init__(self) :
    #     super(alexnet, self).__init__()
    #     self.conv1 = Conv2d(3, 96, (11,11), (4,4), (0,0))
    #     self.conv2 = Conv2d(96, 256, (5,5), (1,1), (2,2))
    #     self.conv3 = Conv2d(256, 384, (3,3), (1,1), (1,1))
    #     self.conv4 = Conv2d(384, 384, (3,3), (1,1), (1,1))
    #     self.conv5 = Conv2d(384, 256, (3,3), (1,1), (1,1))
    #     self.fc1 = Linear(256, 4096)
    #     self.fc2 = Linear(4096, 4096)
    #     self.fc3 = Linear(4096, 1049)
    #     self.mp1 = MaxPool2d((3,3), (2,2))
    #     self.mp2 = MaxPool2d((3,3), (2,2))
    #     self.mp3 = MaxPool2d((3,3), (2,2))

    # def forward(self, x) :
    #     x = F.relu(self.conv1(x))
    #     x = self.mp1(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.mp2(x)
    #     x = F.relu(self.conv3(x))
    #     x = F.relu(self.conv4(x))
    #     x = F.relu(self.conv5(x))
    #     x = self.mp3(x)
    #     x = AdaptiveAvgPool2d(1)(x).squeeze()
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
