import torch

import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#매트릭 (모니터링용 지표)

preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5,(10, ))
print(preds,target)

acc=torchmetrics.functional.accuracy(preds, target,task="multiclass", num_classes=5)
print(acc)
print()


metric = torchmetrics.Accuracy( task="multiclass", num_classes=5)

n_batches =10
for i in range(n_batches):
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = metric(preds, target)
    print(acc)

ac=metric.compute()
print('\n',acc)