import torch

import torch.nn as nn
import torch.nn.functional as F

#신경망 구성

def main():
    #컨볼루션 레이어
    nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1)
    layer = nn.Conv2d(1,20,5,1).to(torch.device('cuda:0'))
    print(layer)

    weight = layer.weight
    print(weight.shape)
    print()

    #풀링 레이어
    #pool = F.max_pool2d(output,2,2)

    #선형 레이어
    # 1d 로 펼쳐줌 



if __name__ == '__main__':
    main()