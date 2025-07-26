import torch

#텐서 조작

x = torch.Tensor([[1,2],[3,4]])
print(x)

print(x[0,0])
print(x[0,1])
print(x[1,0])
print(x[1,1])

#슬라이싱
print(x[:,0])
print(x[:,1])
print(x[0,:])
print(x[1,:])

x=torch.randn(4,5)
print(x)
y= x.view(20)
print(y)
print()

#x.item() 값이 하나만 있을 때에 값을 가져옴
tensor = torch.rand(1,3,3)
print(tensor)
print(tensor.shape)

t = tensor.squeeze()
print(t)
print(t.shape)

t.unsqueeze_(2)
print(t)
print(t.shape)
print()

#stack
x= torch.FloatTensor([1,4])
print(x)
y= torch.FloatTensor([2,5])
print(y)
z= torch.FloatTensor([3,6])
print(z)
print(torch.stack([x,y,z]))
print()

#cat
a= torch.randn(1,3,3)
b= torch.randn(1,3,3)
c= torch.cat((a,b),dim=1)
print(a)
print(b)
print(c.size())

#chunk
tensor = torch.rand(3,6)
print(tensor)
t1,t2,t3=torch.chunk(tensor,3,dim=1)
print(t1)
print(t2)
print(t3)
print()

#split
tensor = torch.rand(3,6)
print(tensor)
t1,t2,t3=torch.chunk(tensor,3,dim=1)
print(t1)
print(t2)
print(t3)

#numpy

a = torch.ones(7)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)
print()

import numpy as np

a = np.ones(7)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
