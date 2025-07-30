import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

import matplotlib.pyplot as plt

#데이터셋
x=torch.randn(200,1)*10
y=x+3*torch.randn(200,1)
plt.scatter(x.numpy(),y.numpy())
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()

#심플 모델
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        pred = self.linear(x)
        return pred

model = LinearRegressionModel()
print(model)
print(list(model.parameters()))

w,b = model.parameters()
w1,b1 = w[0][0].item() , b[0].item()
x1 = np.array([-30,30])
y1 = w1*x1+b1

plt.plot(x1,y1,'r')
plt.scatter(x,y)
plt.grid()
plt.show()

#손실함수 설정
criterion = nn.MSELoss()
#옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=0.001)

#학습
epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()

plt.plot(range(epochs),losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

