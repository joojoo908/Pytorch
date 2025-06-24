import torch
print(torch.__version__)            # PyTorch 버전
print(torch.version.cuda)           # CUDA 버전 (PyTorch가 인식하는)
print(torch.cuda.is_available())    # GPU 사용 가능 여부
print(torch.cuda.get_device_name(0)) # GPU 이름 출력