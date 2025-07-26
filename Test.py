import torch

import torch.nn as nn

#신경망 구성

def main():
    #nn.Linear 계층
    input = torch.randn(128,20)
    print(input)

    m=nn.Linear(20,30)
    print(m)

    output = m(input)
    print(output)
    print(output.size())
    print()

    #nn.conv2d 계층
    input = torch.randn(20,16,50,100)
    print(input.size())
    m=nn.Conv2d(16,33,3,stride=2)
    m=nn.Conv2d(16,33,(3,5),stride=(2,1),
                padding=(4,2))
    m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1),
                  padding=(4, 2), dilation=(3,1))
    print(m)
    output = m(input)
    #print(output)
    print(output.size())
    print()

    #컨볼루션 레이어

if __name__ == '__main__':
    main()