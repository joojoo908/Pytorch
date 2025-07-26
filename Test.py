import torch

#자동미분
a = torch.randn(3, 3)  # 3x3 랜덤 텐서 생성
a = a * 3              # 값들을 3배로 스케일링
print(a)               # 텐서 출력
print(a.requires_grad) # 현재 gradient 추적 여부 (False)

a.requires_grad_(True) # 이제부터 a에 대해 gradient 추적 활성화
print(a.requires_grad) # True 출력

b = (a * a).sum()      # b는 a^2 의 총합 (스칼라)
print(b)               # b 값 출력
print(b.grad_fn)       # b가 어떤 연산으로 만들어졌는지 추적함
print()

#기울기
x=torch.ones(3,3,requires_grad=True)
print(x)
y = x + 5
print(y)

z=y*y
out = z.mean()
print(z,out)

print(out)
out.backward()
print()