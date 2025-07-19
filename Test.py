import torch

#텐서 차원

t0 =torch.Tensor(0)
print(t0.ndim) #1차원
print(t0.shape) #크기
print(t0)
print()

t1 =torch.Tensor([1,2,3])
print(t1.ndim) #1차원
print(t1.shape) #크기
print(t1)
print()

t2 =torch.Tensor([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
print(t2.ndim) #1차원
print(t2.shape) #크기
print(t2)
print()

t3 =torch.Tensor([ [[1,2,3],[4,5,6],[7,8,9]],
                   [[1,2,3],[4,5,6],[7,8,9]],
                   [[1,2,3],[4,5,6],[7,8,9]] ])
print(t3.ndim) #1차원
print(t3.shape) #크기
print(t3)
print()

#4d 텐서는 주로 컬러 이미지에 사용
#5d 텐서는 비디오 데이터에 대표적으로 이용











