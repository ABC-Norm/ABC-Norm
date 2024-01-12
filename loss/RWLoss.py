import torch
import torch.nn as nn

class RWLoss(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super(RWLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight.type(torch.float))

    def forward(self, logits, targets, *args):
        return self.CE(logits, targets), None

