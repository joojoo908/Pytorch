from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch

#import matplotlib.pyplot as plt

def main():
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(1.0,))
    ])

    trainset = datasets.MNIST(root='./', train=True, download=True, transform=mnist_transform)
    testset = datasets.MNIST(root='./', train=False, download=True, transform=mnist_transform)

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2) #멀티프로세싱
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(images.shape, labels.shape)

    torch_image = torch.squeeze(images[0])
    print(torch_image.shape)



#메인을 따로 선언해주어 멀티프로세싱이 메인코드를 다시 실행하여 무한 재귀에 빠져드는 것을 방지함
#데이터 로더가 전역에 있을 때에는 주의할 것
if __name__ == '__main__':
    main()