import torch

import torch.nn as nn
import torch.nn.functional as F

#신경망 구성(레이어로 이루어짐)
#모델 정의

# class Model(nn.Module):
#     def __init__(self,inputs):
#         super(Model, self).__init__()
#         self.layer = nn.Linear(inputs,1)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.layer(x)
#         x = self.activation(x)
#         return(x)

# model = Model(1)
# print(list(model.children()))
# print(list(model.modules()))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.shape[0], -1)
        x=self.layer3(x)
        return x

model = Model()
print(list(model.children()))
print(list(model.modules()))