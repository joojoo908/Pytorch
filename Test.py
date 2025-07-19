import torch

#텐서 연산
import math
a = torch.rand(1,2) *2 -1
print(a)
print(torch.abs(a)) # 절댓값
print(torch.ceil(a)) # 올림
print(torch.floor(a)) # 내림
print(torch.clamp(a,-0.5,0.5)) # 최대치와 최소치 설정
print()

print(a)
print(torch.min(a))       # 최솟값
print(torch.max(a))       # 최댓값
print(torch.mean(a))      # 평균
print(torch.std(a))       # 표준편차
print(torch.prod(a))      # 모든 값의 곱
print(torch.unique(torch.tensor([1,2,3,1,2,2])))  # 중복 제거된 유일한 값들
print()

x = torch.rand(2,2)
print(x)
print(x.max(dim=0))
print(x.max(dim=1))
print()

y = torch.rand(2,2)
print(x)
print(y)
print(torch.add(x,y)) # x+y
result = torch.empty(2,2)
torch.add(x,y,out=result)
print(result)
print()

#in-place 방식
y.add_(x) # y에 x를 더한 값을 y에 저장하여라
print(y)
print()

#내적(dot product)
print(x)
print(y)
print(torch.matmul(x,y))
z = torch.mm(x,y)
print(z)

print(torch.svd(z)) # 행렬 분해 기법









