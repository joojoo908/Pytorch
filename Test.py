import torch
print(torch.__version__)            # PyTorch 버전
print(torch.version.cuda)           # CUDA 버전 (PyTorch가 인식하는)
print(torch.cuda.is_available())    # GPU 사용 가능 여부
print(torch.cuda.get_device_name(0)) # GPU 이름 출력

import numpy as np

#x = torch.empty(4,2) #비운 상태로 초기화
#x = torch.rand(4,2) #무작위로 초기화
#x = torch.zeros(4,2, dtype=torch.long) #0으로 초기화, 사용자가 원하는 데이터 형으로
# x = torch.tensor([3,2.3]) #사용자가 원하는 대로 초기화
# x = x.new_ones(2,4,dtype=torch.double) #원하는 대로 텐서 변경
# x = torch.randn_like(x,dtype=torch.float) #x와 같은 크기 무작위로 채워진 텐서
# print(x)
# print(x.size()) # 텐서 사이즈

#텐서 타입
ft = torch.FloatTensor([1,2,3])
print(ft)
print(ft.dtype)
it = torch.IntTensor([1,2,3])
print(it)
print(it.dtype)

#ft.short(), ft.int(), ft.long() 등으로 데이터 타입 변경

x = torch.randn(1)
print("x =" ,x)
print(x.dtype)

print()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 쿠다 혹은 cpu 이용
print(device)
y = torch.ones_like(x,device=device)
print("y: ", y)
x= x.to(device)
print("x: ", x)
z=x+y
print("z: ", z)
print(z.to('cpu',torch.double))